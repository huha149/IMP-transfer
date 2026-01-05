from __future__ import annotations

import json
import os
from typing import Dict, Any

import optuna
import torch

from ..training.train_cv import train_with_cross_validation


def objective(
    trial: optuna.Trial,
    hyper_params_grid: Dict[str, Dict[str, list]],
    config_params: Dict[str, Any],
    output_dir: str,
) -> float:
    """Optuna objective for CV AUC.

    说明：
    - 该函数会为每个 trial 创建一个独立输出目录
    - 保存：best_model_state_dict.pt、fold_best_val_aucs.json、hyperparameters.json
    """
    model_params = {
        k: trial.suggest_categorical(k, v)
        for k, v in hyper_params_grid["model_params"].items()
    }
    optimizer_paras = {
        k: trial.suggest_categorical(k, v)
        for k, v in hyper_params_grid["optimizer_paras"].items()
    }
    early_stopping_params = {
        k: trial.suggest_categorical(k, v)
        for k, v in hyper_params_grid["early_stopping_params"].items()
    }
    criterion_params = {
        k: trial.suggest_categorical(k, v)
        for k, v in hyper_params_grid["criterion_params"].items()
    }

    trial_dir = os.path.join(output_dir, "trials", f"trial_{trial.number:04d}")
    cfg = dict(config_params)
    cfg["output_dir"] = trial_dir

    best_model, avg_auc, fold_best_val_aucs = train_with_cross_validation(
        optimizer_paras=optimizer_paras,
        early_stopping_params=early_stopping_params,
        model_params=model_params,
        criterion_params=criterion_params,
        config_params=cfg,
    )

    os.makedirs(trial_dir, exist_ok=True)

    # 保存最优模型（state_dict）
    torch.save(best_model.state_dict(), os.path.join(trial_dir, "best_model_state_dict.pt"))

    # 保存该 trial 的指标与超参数
    with open(os.path.join(trial_dir, "fold_best_val_aucs.json"), "w", encoding="utf-8") as f:
        json.dump({"fold_best_val_aucs": fold_best_val_aucs, "avg_auc": avg_auc}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(trial_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_params": model_params,
                "optimizer_paras": optimizer_paras,
                "early_stopping_params": early_stopping_params,
                "criterion_params": criterion_params,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return float(avg_auc)
