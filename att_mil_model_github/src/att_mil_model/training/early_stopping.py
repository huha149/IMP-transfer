from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    verbose: bool = False
    delta: float = 0.0
    save_dir: str = "path/to/output/fold_1"
    filename: str = "checkpoint.pt"


class EarlyStopping:
    """Early stopping utility.

    - 监控验证集 loss，当连续 patience 个 epoch 无改进时停止训练
    - 当验证 loss 改善时保存 checkpoint（默认保存 state_dict，推荐用于 GitHub 公开仓库）

    checkpoint 格式：
    {
        "model_state_dict": ...,
        "epoch": int,
        "val_loss": float,
        "extra": dict (可选)
    }
    """

    def __init__(self, config: EarlyStoppingConfig):
        self.patience = config.patience
        self.verbose = config.verbose
        self.delta = config.delta
        self.save_dir = config.save_dir
        self.filename = config.filename

        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.filename)

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(
        self,
        val_loss: float,
        model: torch.nn.Module,
        epoch: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, epoch, extra)
            return

        if score < (self.best_score + self.delta):
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self._save_checkpoint(val_loss, model, epoch, extra)
        self.counter = 0

    def _save_checkpoint(
        self,
        val_loss: float,
        model: torch.nn.Module,
        epoch: int,
        extra: Optional[Dict[str, Any]],
    ) -> None:
        if self.verbose:
            print(f"[EarlyStopping] Val loss improved: {self.val_loss_min:.6f} -> {val_loss:.6f}. Saving...")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "extra": extra or {},
        }
        torch.save(ckpt, self.save_path)
        self.val_loss_min = val_loss
