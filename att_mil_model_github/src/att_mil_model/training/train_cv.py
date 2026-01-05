from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from ..models.att_mil_model import att_mil_model
from ..utils.plotting import save_loss_curve
from .early_stopping import EarlyStopping, EarlyStoppingConfig


@dataclass
class TrainConfig:
    num_epochs: int = 50
    device: str = "cuda:0"
    output_dir: str = "path/to/output"
    n_splits: int = 5
    seed: int = 149


def _compute_multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUC for binary/multiclass using sklearn.roc_auc_score."""
    num_classes = y_prob.shape[1]
    if num_classes == 2:
        # 二分类：取正类概率
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    # 多分类：y_prob shape (N, C)
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr"))


def train_with_cross_validation(
    optimizer_paras: Dict,
    early_stopping_params: Dict,
    model_params: Dict,
    criterion_params: Dict,
    config_params: Dict,
) -> Tuple[att_mil_model, float, List[float]]:
    """5-fold CV training with early stopping.

    Parameters
    ----------
    optimizer_paras:
        {"learning_rate": float, "T_max": int, "eta_min": float}
    early_stopping_params:
        {"patience": int, "verbose": bool, "delta": float}
    model_params:
        att_mil_model 初始化参数
    criterion_params:
        例如：{"label_smoothing": float}
    config_params:
        {"dataset": Dataset, "num_epochs": int, "device": str, "output_dir": str, "n_splits": int}

    Returns
    -------
    best_model: 验证 AUC 最佳的模型（加载最佳 fold 的 state_dict）
    avg_auc: 各 fold 最佳 AUC 平均值
    fold_best_val_aucs: 每个 fold 的最佳 AUC
    """
    dataset = config_params["dataset"]
    num_epochs = int(config_params.get("num_epochs", 50))
    device = str(config_params.get("device", "cuda:0"))
    output_dir = str(config_params.get("output_dir", "path/to/output"))
    n_splits = int(config_params.get("n_splits", 5))
    seed = int(config_params.get("seed", 149))

    os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_best_val_aucs: List[float] = []
    best_auc = -1.0
    best_model_state: Optional[dict] = None

    # 判断任务类型
    task = int(model_params.get("Task", 2))
    is_classification = (task == 2)

    label_smoothing = float(criterion_params.get("label_smoothing", 0.0))

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        print(f"[CV] Fold {fold}/{n_splits}")

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        model = att_mil_model(**model_params).to(device)

        if is_classification:
            # CrossEntropyLoss 需要 logits（不要提前 softmax）
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            criterion = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=float(optimizer_paras["learning_rate"]))
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(optimizer_paras["T_max"]),
            eta_min=float(optimizer_paras["eta_min"]),
        )

        es_config = EarlyStoppingConfig(
            patience=int(early_stopping_params.get("patience", 10)),
            verbose=bool(early_stopping_params.get("verbose", False)),
            delta=float(early_stopping_params.get("delta", 0.0)),
            save_dir=fold_dir,
            filename="checkpoint.pt",
        )
        early_stopping = EarlyStopping(es_config)

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_fold_auc = -1.0
        best_fold_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            # ---------------------------
            # Train
            # ---------------------------
            model.train()
            total_train_loss = 0.0

            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                out = model(data).logits  # (1, C) or (1, 1)

                if is_classification:
                    loss = criterion(out, target)
                else:
                    # 回归：target 转 float，并与 out 对齐形状
                    target_f = target.float().view_as(out)
                    loss = criterion(out, target_f)

                loss.backward()
                optimizer.step()
                total_train_loss += float(loss.item())

            scheduler.step()
            avg_train_loss = total_train_loss / max(len(train_loader), 1)
            train_losses.append(avg_train_loss)

            # ---------------------------
            # Validate
            # ---------------------------
            model.eval()
            total_val_loss = 0.0
            y_true: List[int] = []
            y_prob: List[np.ndarray] = []

            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = target.to(device)

                    logits = model(data).logits

                    if is_classification:
                        loss = criterion(logits, target)
                        probs = F.softmax(logits, dim=1).detach().cpu().numpy()  # (1, C)
                        y_true.extend(target.detach().cpu().numpy().reshape(-1).tolist())
                        y_prob.append(probs.reshape(1, -1))
                    else:
                        target_f = target.float().view_as(logits)
                        loss = criterion(logits, target_f)

                    total_val_loss += float(loss.item())

            avg_val_loss = total_val_loss / max(len(val_loader), 1)
            val_losses.append(avg_val_loss)

            # AUC（分类任务）
            if is_classification:
                y_true_arr = np.array(y_true, dtype=np.int64)
                y_prob_arr = np.vstack(y_prob)  # (N, C)
                val_auc = _compute_multiclass_auc(y_true_arr, y_prob_arr)
            else:
                val_auc = float("nan")

            if is_classification and (avg_val_loss < best_fold_val_loss - es_config.delta):
                best_fold_val_loss = avg_val_loss
                best_fold_auc = val_auc

            if is_classification:
                print(
                    f"[Fold {fold}] Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}"
                )
            else:
                print(
                    f"[Fold {fold}] Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                )

            # 保存 checkpoint（val_loss 作为早停指标）
            extra = {"model_params": model_params, "optimizer_paras": optimizer_paras, "criterion_params": criterion_params}
            early_stopping(avg_val_loss, model, epoch=epoch, extra=extra)

            if early_stopping.early_stop:
                print(f"[CV] Early stopping triggered at Fold {fold}.")
                break

        # 保存曲线
        save_loss_curve(train_losses, os.path.join(fold_dir, "train_loss.pdf"), title=f"Fold {fold} Train Loss")
        save_loss_curve(val_losses, os.path.join(fold_dir, "val_loss.pdf"), title=f"Fold {fold} Val Loss")

        if is_classification:
            fold_best_val_aucs.append(best_fold_auc)
            if best_fold_auc > best_auc:
                best_auc = best_fold_auc
                best_model_state = torch.load(os.path.join(fold_dir, "checkpoint.pt"), map_location="cpu")["model_state_dict"]

    if not fold_best_val_aucs:
        raise RuntimeError("No AUC values were computed. Please check Task/model_params and dataset labels.")

    avg_auc = float(np.mean(fold_best_val_aucs))
    print(f"[CV] Average Val AUC across {n_splits} folds: {avg_auc:.4f}")

    # 构建并加载最佳模型
    best_model = att_mil_model(**model_params).to(device)
    if best_model_state is not None:
        best_model.load_state_dict(best_model_state)
    return best_model, avg_auc, fold_best_val_aucs


def evaluate_cross_validation_models(
    test_loader: DataLoader,
    fold_dir: str,
    device: str,
    model_params: Dict,
    n_splits: int = 5,
) -> List[float]:
    """Evaluate saved fold checkpoints on a given loader and return AUC list."""
    task = int(model_params.get("Task", 2))
    if task != 2:
        raise ValueError("evaluate_cross_validation_models currently supports classification Task=2 only.")

    test_aucs: List[float] = []

    for fold in range(1, n_splits + 1):
        ckpt_path = os.path.join(fold_dir, f"fold_{fold}", "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = att_mil_model(**model_params).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        y_true: List[int] = []
        y_prob: List[np.ndarray] = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                logits = model(data).logits
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()  # (1, C)

                y_true.extend(target.detach().cpu().numpy().reshape(-1).tolist())
                y_prob.append(probs.reshape(1, -1))

        y_true_arr = np.array(y_true, dtype=np.int64)
        y_prob_arr = np.vstack(y_prob)
        auc = _compute_multiclass_auc(y_true_arr, y_prob_arr)
        test_aucs.append(auc)
        print(f"[Eval] Fold {fold} AUC: {auc:.4f}")

    return test_aucs
