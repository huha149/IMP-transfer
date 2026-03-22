import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.models.attention_mil import AttentionMILModel
from src.training.early_stopping import EarlyStopping
from src.utils.save_loss_curve import save_loss_curve


def train_with_cross_validation(
    optimizer_params: dict,
    early_stopping_params: dict,
    model_params: dict,
    criterion_params: dict,
    config_params: dict,
):
    """
    Train a model with K-fold cross-validation.

    Returns:
        best_model: Model from the best-performing fold.
        avg_auc: Mean validation AUC across folds.
        fold_best_val_aucs: Best validation AUC from each fold.
    """
    dataset = config_params["dataset"]
    num_epochs = config_params["num_epochs"]
    device = config_params["device"]
    output_dir = config_params["output_dir"]
    n_splits = config_params["n_splits"]
    label_smoothing = criterion_params["label_smoothing"]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=149)
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

        model = AttentionMILModel(**model_params).to(device)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params["learning_rate"])
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

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output, _, _ = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0.0
            val_targets = []
            val_preds = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output, _, _ = model(data)
                    probs = F.softmax(output, dim=1)
                    loss = criterion(output, target)
                    total_val_loss += loss.item()

                    val_targets.extend(target.cpu().numpy().reshape(-1))
                    val_preds.extend(probs.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            all_predictions = np.vstack(val_preds)
            val_auc = roc_auc_score(
                np.array(val_targets),
                np.squeeze(np.array(all_predictions)),
                multi_class="ovr",
            )

            if avg_val_loss < best_fold_val_loss - fold_early_stopping_params["delta"]:
                best_fold_val_loss = avg_val_loss
                best_fold_auc = val_auc

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
    Evaluate saved fold checkpoints on a dataset and return one AUC per fold.
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
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _, _ = model(data)
                probs = torch.softmax(output, dim=1)

                all_targets.extend(target.cpu().numpy().reshape(-1))
                all_probs.extend(probs.cpu().numpy())

        all_probs = np.vstack(all_probs)
        test_auc = roc_auc_score(
            np.array(all_targets),
            np.squeeze(np.array(all_probs)),
            multi_class="ovr",
        )
        test_aucs.append(test_auc)
        print(f"Fold {fold} test AUC: {test_auc:.4f}")

    print("All folds evaluated.")
    return test_aucs
