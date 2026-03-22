import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.models.attention_mil import AttentionMILModel
from src.training.early_stopping import EarlyStopping


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


def train_with_cross_validation(
    optimizer_params: dict,
    early_stopping_params: dict,
    model_params: dict,
    config_params: dict,
):
    """
    Train the model using K-fold cross-validation.

    Returns:
        best_rmse_model: Best model by validation RMSE.
        best_mape_model: Best model by validation MAPE.
        best_auc_model: Best model by derived validation AUC.
        fold_best_rmses: Best validation RMSE for each fold.
        fold_best_mapes: Best validation MAPE for each fold.
        fold_best_aucs: Best derived validation AUC for each fold.
    """
    dataset = config_params["dataset"]
    num_epochs = config_params["num_epochs"]
    device = config_params["device"]
    output_dir = config_params["output_dir"]
    n_splits = config_params["n_splits"]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config_params["seed"])
    best_rmse = float("inf")
    best_mape = float("inf")
    best_auc = float("-inf")
    best_rmse_model_state = None
    best_mape_model_state = None
    best_auc_model_state = None
    fold_best_rmses = []
    fold_best_mapes = []
    fold_best_aucs = []

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

        criterion = torch.nn.MSELoss()
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

        fold_best_loss = float("inf")
        fold_best_rmse = None
        fold_best_mape = None
        fold_best_auc = None
        fold_best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            if train_mode in ["pretrained_freeze_att", "random_freeze_att"]:
                model.attention_net.eval()

            total_train_loss = 0.0
            for data, target, _ in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output, _, _ = model(data)
                loss = criterion(output.squeeze(), target.squeeze())
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0.0
            val_targets = []
            val_predictions = []

            with torch.no_grad():
                for data, target, _ in val_loader:
                    data, target = data.to(device), target.to(device)
                    output, _, _ = model(data)
                    loss = criterion(output.squeeze(), target.squeeze())
                    total_val_loss += loss.item()
                    val_targets.extend(target.cpu().numpy().reshape(-1))
                    val_predictions.extend(output.cpu().numpy().reshape(-1))

            avg_val_loss = total_val_loss / len(val_loader)
            val_targets = np.array(val_targets)
            val_predictions = np.array(val_predictions)

            val_rmse = float(np.sqrt(mean_squared_error(val_targets, val_predictions)))
            denom = np.where(np.abs(val_targets) < 1e-8, 1.0, np.abs(val_targets))
            val_mape = float(np.mean(np.abs((val_targets - val_predictions) / denom)))

            threshold = np.median(val_targets)
            val_targets_binarized = (val_targets > threshold).astype(int)
            val_auc = float(roc_auc_score(val_targets_binarized, val_predictions))

            print(
                f"Fold {fold}, Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}, "
                f"RMSE: {val_rmse:.4f}, "
                f"MAPE: {val_mape:.4f}, "
                f"AUC: {val_auc:.4f}"
            )

            if avg_val_loss < fold_best_loss - fold_early_stopping_params["delta"]:
                fold_best_loss = avg_val_loss
                fold_best_rmse = val_rmse
                fold_best_mape = val_mape
                fold_best_auc = val_auc
                fold_best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(fold_best_model_state, os.path.join(fold_dir, "best_model_state.pt"))
                np.save(os.path.join(fold_dir, "val_predictions.npy"), val_predictions)
                np.save(os.path.join(fold_dir, "val_targets.npy"), val_targets)

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered in fold {fold}.")
                break

        fold_best_rmses.append(fold_best_rmse)
        fold_best_mapes.append(fold_best_mape)
        fold_best_aucs.append(fold_best_auc)

        if fold_best_rmse is not None and fold_best_rmse < best_rmse:
            best_rmse = fold_best_rmse
            best_rmse_model_state = fold_best_model_state

        if fold_best_mape is not None and fold_best_mape < best_mape:
            best_mape = fold_best_mape
            best_mape_model_state = fold_best_model_state

        if fold_best_auc is not None and fold_best_auc > best_auc:
            best_auc = fold_best_auc
            best_auc_model_state = fold_best_model_state

    best_rmse_model = AttentionMILModel(**model_params).to(device)
    best_rmse_model.load_state_dict(best_rmse_model_state)

    best_mape_model = AttentionMILModel(**model_params).to(device)
    best_mape_model.load_state_dict(best_mape_model_state)

    best_auc_model = AttentionMILModel(**model_params).to(device)
    best_auc_model.load_state_dict(best_auc_model_state)

    return (
        best_rmse_model,
        best_mape_model,
        best_auc_model,
        fold_best_rmses,
        fold_best_mapes,
        fold_best_aucs,
    )


def sanitize_filename(name: str) -> str:
    """
    Replace unsafe characters in file names.
    """
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(name))


def _round3(x) -> float:
    return float(f"{x:.3f}")


@torch.no_grad()
def evaluate_cv_best_models_on_test(
    output_dir: str,
    dataset,
    dataset_name: str,
    model_params: dict,
    device: str,
    n_folds: int = 5,
    cv_subdir: str = "cross_validation",
    save_attention_per_fold: bool = True,
):
    """
    Evaluate saved fold checkpoints on a regression dataset.
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

    rows = []
    all_labels = []
    all_predictions_mean = []
    all_predictions_per_fold = [[] for _ in range(n_folds)]

    for batch in loader:
        bag_features = batch[0].to(device)
        bag_label = batch[1].to(device)
        bag_id = batch[2]

        bag_id_str = str(bag_id[0]) if isinstance(bag_id, (list, tuple)) else str(bag_id)
        bag_id_safe = sanitize_filename(bag_id_str)

        fold_predictions = []
        fold_attentions = []

        for fold_idx in range(n_folds):
            model = AttentionMILModel(**model_params).to(device)
            model.load_state_dict(fold_states[fold_idx], strict=True)
            model.eval()

            logits, _, attention_scores = model(bag_features)
            prediction = float(logits.squeeze().cpu().item())
            fold_predictions.append(prediction)
            fold_attentions.append(attention_scores.cpu())

        prediction_mean = float(np.mean(fold_predictions))
        label_value = float(bag_label.squeeze().cpu().item())

        attention_payload = {
            "bag_id": bag_id_str,
            "label": label_value,
            "pred_mean": prediction_mean,
        }
        if save_attention_per_fold:
            attention_payload["attention_per_fold"] = fold_attentions
            attention_payload["pred_per_fold"] = fold_predictions
        else:
            attention_stack = torch.stack([attention.squeeze(0) for attention in fold_attentions], dim=0)
            attention_payload["attention_mean"] = attention_stack.mean(dim=0)
            attention_payload["pred_per_fold"] = fold_predictions

        torch.save(attention_payload, os.path.join(attention_dir, f"{bag_id_safe}.pt"))

        row = {
            "bag_id": bag_id_str,
            "label": label_value,
            "pred_mean": prediction_mean,
        }
        for i, prediction in enumerate(fold_predictions, start=1):
            row[f"pred_fold_{i}"] = prediction
        rows.append(row)

        all_labels.append(label_value)
        all_predictions_mean.append(prediction_mean)
        for fold_idx in range(n_folds):
            all_predictions_per_fold[fold_idx].append(fold_predictions[fold_idx])

    rmse_mean = float(np.sqrt(mean_squared_error(all_labels, all_predictions_mean)))
    pearson_r_mean, pearson_p_mean = pearsonr(all_labels, all_predictions_mean)

    fold_metrics = []
    for fold_idx in range(n_folds):
        predictions = all_predictions_per_fold[fold_idx]
        rmse_fold = float(np.sqrt(mean_squared_error(all_labels, predictions)))
        pearson_r_fold, pearson_p_fold = pearsonr(all_labels, predictions)
        fold_metrics.append(
            {
                "fold": fold_idx + 1,
                "rmse": _round3(rmse_fold),
                "pearson_r": _round3(pearson_r_fold),
                "pearson_p": float(pearson_p_fold),
            }
        )

    metrics = {
        "n_samples": int(len(all_labels)),
        "mean_pred_metrics": {
            "rmse": _round3(rmse_mean),
            "pearson_r": _round3(pearson_r_mean),
            "pearson_p": float(pearson_p_mean),
        },
        "per_fold_metrics": fold_metrics,
    }

    df = pd.DataFrame(rows)
    df.to_csv(prediction_csv_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"[OK] saved predictions: {prediction_csv_path}")
    print(f"[OK] saved metrics: {metrics_path}")
    print(f"[OK] saved attentions: {attention_dir}")
    print(
        f"[Test-MeanPred] RMSE={rmse_mean:.3f}, "
        f"Pearson r={pearson_r_mean:.3f} (p={pearson_p_mean:.3e})"
    )

    return metrics, df
