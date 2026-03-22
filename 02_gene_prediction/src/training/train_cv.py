import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.models.attention_mil import AttentionMILModel
from src.training.early_stopping import EarlyStopping
from src.training.save_loss_curve import save_loss_curve


def freeze_module(module: nn.Module) -> None:
    """
    Freeze all parameters in a module.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def reset_module_parameters(module: nn.Module) -> None:
    """
    Reset parameters for submodules that implement `reset_parameters`.
    """
    for submodule in module.modules():
        if hasattr(submodule, "reset_parameters"):
            submodule.reset_parameters()


def load_pretrained_attention(model: nn.Module, pretrained_path: str, device: str) -> nn.Module:
    """
    Load only the attention block weights from a pretrained IMP model checkpoint.
    """
    checkpoint = torch.load(pretrained_path, map_location=device)
    state = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

    attention_state = {k: v for k, v in state.items() if k.startswith("attention_net.")}
    message = model.load_state_dict(attention_state, strict=False)
    print(f"[load_pretrained_attention] {message}")
    return model


def build_model_by_mode(model_params: dict, config_params: dict, device: str):
    """
    Build a model according to the selected training strategy.
    """
    train_mode = config_params.get("train_mode", "scratch")
    pretrained_path = config_params.get("pretrained_path")

    model = AttentionMILModel(**model_params).to(device)

    if train_mode == "scratch":
        return model, train_mode

    if train_mode in ["pretrained_finetune", "pretrained_freeze_att"]:
        if pretrained_path is None:
            raise ValueError(
                f"train_mode={train_mode} requires config_params['pretrained_path']"
            )
        model = load_pretrained_attention(model, pretrained_path, device)

    if train_mode == "random_freeze_att":
        reset_module_parameters(model.attention_net)

    reset_module_parameters(model.projector)
    reset_module_parameters(model.prediction)

    if train_mode in ["pretrained_freeze_att", "random_freeze_att"]:
        freeze_module(model.attention_net)

    return model, train_mode


def count_positive_negative_samples(dataloader) -> tuple[int, int]:
    """
    Count positive and negative samples in a dataloader.

    Returns:
        (num_positive, num_negative)
    """
    num_positive = 0
    num_negative = 0

    for _, label, _ in dataloader:
        label_value = int(torch.as_tensor(label).view(-1)[0].item())
        if label_value == 1:
            num_positive += 1
        elif label_value == 0:
            num_negative += 1

    return num_positive, num_negative


def compute_weighted_ce_loss(logits: torch.Tensor, target: torch.Tensor, num_positive: int, num_negative: int):
    """
    Compute weighted cross-entropy loss for binary classification.

    Class weights are defined as:
    - weight for class 0 = 1 / num_negative
    - weight for class 1 = 1 / num_positive

    This design highlights the importance of weighted CE loss for imbalanced gene prediction.
    """
    if num_positive <= 0 or num_negative <= 0:
        raise ValueError("Both positive and negative sample counts must be greater than zero.")

    weights = torch.tensor(
        [1.0 / num_negative, 1.0 / num_positive],
        device=logits.device,
        dtype=torch.float32,
    )
    return F.cross_entropy(input=logits, target=target, weight=weights)


def train_with_cross_validation(
    optimizer_params: dict,
    early_stopping_params: dict,
    model_params: dict,
    config_params: dict,
):
    """
    Train the model using K-fold cross-validation.

    Returns:
        best_model: Best-performing fold model.
        avg_auc: Mean validation AUC across folds.
        fold_best_val_aucs: Best validation AUC from each fold.
    """
    dataset = config_params["dataset"]
    num_epochs = config_params["num_epochs"]
    device = config_params["device"]
    output_dir = config_params["output_dir"]
    n_splits = config_params["n_splits"]
    use_weighted_ce = config_params["ce_weight"]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config_params["seed"])
    fold_best_val_aucs = []
    best_auc = -1.0
    best_fold = -1
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        print(f"Starting fold {fold}/{n_splits}")

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False)

        model, train_mode = build_model_by_mode(model_params, config_params, device)
        print(f"[Fold {fold}] train_mode = {train_mode}")

        trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=optimizer_params["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=optimizer_params["T_max"],
            eta_min=optimizer_params["eta_min"],
        )

        fold_early_stopping_params = dict(early_stopping_params)
        fold_early_stopping_params["save_path"] = fold_dir
        early_stopping = EarlyStopping(**fold_early_stopping_params)

        train_losses = []
        val_losses = []
        best_fold_val_loss = float("inf")
        best_fold_auc = -1.0

        if use_weighted_ce:
            train_positive, train_negative = count_positive_negative_samples(train_loader)
            print(
                "[Weighted CE] train split class counts - "
                f"positive: {train_positive}, negative: {train_negative}"
            )

        for epoch in range(num_epochs):
            model.train()
            if train_mode in ["pretrained_freeze_att", "random_freeze_att"]:
                model.attention_net.eval()

            total_train_loss = 0.0
            for data, target, _ in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                logits, _, _ = model(data)

                if use_weighted_ce:
                    loss = compute_weighted_ce_loss(
                        logits=logits,
                        target=target,
                        num_positive=train_positive,
                        num_negative=train_negative,
                    )
                else:
                    loss = F.cross_entropy(input=logits, target=target)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0.0
            val_targets = []
            val_probs = []

            if use_weighted_ce:
                val_positive, val_negative = count_positive_negative_samples(val_loader)
                print(
                    "[Weighted CE] validation split class counts - "
                    f"positive: {val_positive}, negative: {val_negative}"
                )

            with torch.no_grad():
                for data, target, _ in val_loader:
                    data, target = data.to(device), target.to(device)
                    logits, _, _ = model(data)

                    if use_weighted_ce:
                        loss = compute_weighted_ce_loss(
                            logits=logits,
                            target=target,
                            num_positive=val_positive,
                            num_negative=val_negative,
                        )
                    else:
                        loss = F.cross_entropy(input=logits, target=target)

                    probs = F.softmax(logits, dim=1)
                    total_val_loss += loss.item()
                    val_targets.extend(target.cpu().numpy().reshape(-1))
                    val_probs.extend(probs.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            all_predictions = np.vstack(val_probs)
            positive_class_probs = all_predictions[:, 1]
            val_auc = roc_auc_score(np.array(val_targets), positive_class_probs)

            if avg_val_loss < best_fold_val_loss - fold_early_stopping_params["delta"]:
                best_fold_val_loss = avg_val_loss
                best_fold_auc = val_auc
                best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(best_fold_state, os.path.join(fold_dir, "best_model_state.pt"))

            print(
                f"Fold {fold}, Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}, "
                f"Validation AUC: {val_auc:.4f}"
            )

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered in fold {fold}.")
                break

        save_loss_curve(train_losses, os.path.join(fold_dir, f"fold_{fold}_train_loss.pdf"))
        save_loss_curve(val_losses, os.path.join(fold_dir, f"fold_{fold}_val_loss.pdf"))

        if best_fold_auc > best_auc:
            best_auc = best_fold_auc
            best_fold = fold
            best_model_state = model.state_dict()

        fold_best_val_aucs.append(best_fold_auc)

    avg_auc = float(np.mean(fold_best_val_aucs))
    print(f"Average validation AUC across {n_splits} folds: {avg_auc:.4f}")
    print(f"Best model from fold {best_fold} with validation AUC: {best_auc:.4f}")

    best_model = AttentionMILModel(**model_params).to(device)
    best_model.load_state_dict(best_model_state)
    return best_model, avg_auc, fold_best_val_aucs


def evaluate_cross_validation_models(test_loader, fold_dir: str, device: str, num_folds: int = 5):
    """
    Load fold checkpoints and evaluate test AUC for each fold.
    """
    test_aucs = []

    for fold in range(1, num_folds + 1):
        print(f"Evaluating fold {fold} model...")
        model_path = os.path.join(fold_dir, f"fold_{fold}", "checkpoint.pt")
        model = torch.load(model_path).to(device)
        model.eval()

        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target, _ in test_loader:
                data, target = data.to(device), target.to(device)
                logits, _, _ = model(data)
                probs = torch.softmax(logits, dim=1)
                all_targets.extend(target.cpu().numpy().reshape(-1))
                all_probs.extend(probs.cpu().numpy())

        all_probs = np.vstack(all_probs)
        positive_class_probs = all_probs[:, 1]
        test_auc = roc_auc_score(np.array(all_targets), positive_class_probs)
        test_aucs.append(test_auc)
        print(f"Fold {fold} test AUC: {test_auc:.4f}")

    print("All folds evaluated.")
    return test_aucs


def sanitize_filename(name: str) -> str:
    """
    Replace unsafe characters in file names.
    """
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(name))


@torch.no_grad()
def evaluate_cv_best_models_on_test_cls(
    output_dir: str,
    dataset,
    dataset_name: str,
    model_params: dict,
    device: str,
    n_folds: int = 5,
    cv_subdir: str = "cross_validation",
    save_attention_per_fold: bool = True,
    positive_class_index: int = 1,
    threshold: float = 0.5,
):
    """
    Evaluate the best state from each fold on a dataset.

    Outputs:
    - per-fold positive class probabilities
    - mean ensemble probability
    - AUC for each fold
    - ensemble AUC
    - saved attention tensors per bag
    """
    os.makedirs(output_dir, exist_ok=True)
    prediction_csv_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    metrics_path = os.path.join(output_dir, f"{dataset_name}_metrics.json")
    attention_dir = os.path.join(output_dir, f"{dataset_name}_attentions")
    os.makedirs(attention_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    fold_states = []
    for fold in range(1, n_folds + 1):
        state_path = os.path.join(output_dir, cv_subdir, f"fold_{fold}", "best_model_state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing fold checkpoint: {state_path}")
        fold_states.append(torch.load(state_path, map_location="cpu"))

    probs_per_fold_all = [[] for _ in range(n_folds)]
    labels_all = []
    rows = []
    probs_mean_all = []

    for batch in loader:
        bag_features = batch[0].to(device)
        bag_label = batch[1]
        bag_id = batch[2]

        bag_id_str = str(bag_id[0]) if isinstance(bag_id, (list, tuple)) else str(bag_id)
        bag_id_safe = sanitize_filename(bag_id_str)

        label_value = int(torch.as_tensor(bag_label).view(-1)[0].item())
        labels_all.append(label_value)

        fold_probs = []
        fold_attentions = []

        for fold_idx in range(n_folds):
            model = AttentionMILModel(**model_params).to(device)
            model.load_state_dict(fold_states[fold_idx], strict=True)
            model.eval()

            logits, _, attention_scores = model(bag_features)
            probs = F.softmax(logits, dim=1)
            prob_positive = float(probs[0, positive_class_index].cpu().item())

            fold_probs.append(prob_positive)
            probs_per_fold_all[fold_idx].append(prob_positive)
            fold_attentions.append(attention_scores.cpu())

        prob_mean = float(np.mean(fold_probs))
        probs_mean_all.append(prob_mean)
        pred_class = int(prob_mean >= threshold)

        attention_payload = {
            "bag_id": bag_id_str,
            "label": label_value,
            "prob_pos_mean": prob_mean,
            "pred_class": pred_class,
        }

        if save_attention_per_fold:
            attention_payload["attention_per_fold"] = fold_attentions
            attention_payload["prob_pos_per_fold"] = fold_probs
        else:
            attention_stack = torch.stack([attention.squeeze(0) for attention in fold_attentions], dim=0)
            attention_payload["attention_mean"] = attention_stack.mean(dim=0)
            attention_payload["prob_pos_per_fold"] = fold_probs

        torch.save(attention_payload, os.path.join(attention_dir, f"{bag_id_safe}.pt"))

        row = {
            "bag_id": bag_id_str,
            "label": label_value,
            "prob_pos_mean": prob_mean,
            "pred_class": pred_class,
        }
        for i, probability in enumerate(fold_probs, start=1):
            row[f"prob_pos_fold_{i}"] = probability

        rows.append(row)

    y_true = np.array(labels_all, dtype=np.int64)

    auc_per_fold = []
    for fold_idx in range(n_folds):
        y_score = np.array(probs_per_fold_all[fold_idx], dtype=np.float32)
        auc_value = float(roc_auc_score(y_true, y_score))
        auc_per_fold.append(auc_value)

    auc_ensemble = float(roc_auc_score(y_true, np.array(probs_mean_all, dtype=np.float32)))
    auc_mean_of_folds = float(np.mean(auc_per_fold))

    metrics = {
        "n_samples": int(len(labels_all)),
        "auc_ensemble_meanprob": round(auc_ensemble, 3),
        "auc_per_fold": [round(x, 3) for x in auc_per_fold],
        "auc_mean_of_folds": round(auc_mean_of_folds, 3),
        "positive_class_index": int(positive_class_index),
        "threshold_for_pred_class": float(threshold),
    }

    df = pd.DataFrame(rows)
    df.to_csv(prediction_csv_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"[OK] saved predictions: {prediction_csv_path}")
    print(f"[OK] saved metrics: {metrics_path}")
    print(f"[OK] saved attentions: {attention_dir}")
    print(
        f"[Test] AUC ensemble(mean prob)={metrics['auc_ensemble_meanprob']}, "
        f"mean fold AUC={metrics['auc_mean_of_folds']}"
    )

    return metrics, df
