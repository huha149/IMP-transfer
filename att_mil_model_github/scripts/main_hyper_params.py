"""Run Optuna hyperparameter search for att_mil_model.

重要：
- 本脚本不包含任何真实路径，请通过命令行参数传入：
  例如：
    python scripts/main_hyper_params.py \
        --train_csv path/to/labels/train.csv \
        --train_feature_dir path/to/features/train \
        --test_csv path/to/labels/test.csv \
        --test_feature_dir path/to/features/test \
        --output_dir path/to/output \
        --device cuda:0

目录约定（示例）：
- feature_dir 下每个 bag 一个 pt 文件：{bag_id}.pt
- CSV 至少包含：bag_id 列（默认第 0 列）与 slide_label 列（可用 --label_col 指定）

该脚本会输出：
- output_dir/trials/trial_xxxx/...（每个 trial 的交叉验证结果与 checkpoint）
- output_dir/best_params.json
- output_dir/test_auc_boxplot.pdf
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from att_mil_model.data.bags_dataset import BagsDataset_IMP
from att_mil_model.tuning.hyper_params_optimizer import objective
from att_mil_model.training.train_cv import evaluate_cross_validation_models
from att_mil_model.utils.plotting import plot_boxplot


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="path/to/labels/train.csv")
    p.add_argument("--train_feature_dir", type=str, required=True, help="path/to/features/train")
    p.add_argument("--test_csv", type=str, required=True, help="path/to/labels/test.csv")
    p.add_argument("--test_feature_dir", type=str, required=True, help="path/to/features/test")
    p.add_argument("--output_dir", type=str, required=True, help="path/to/output")
    p.add_argument("--label_col", type=str, default="slide_label", help="Label column in CSV")
    p.add_argument("--n_sample", type=int, default=None, help="Optional: load first N samples")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=149)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--n_splits", type=int, default=5)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_jobs", type=int, default=1)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------
    # Dataset & DataLoader
    # -----------------------
    train_dataset = BagsDataset_IMP(
        csv_file=args.train_csv,
        feature_dir=args.train_feature_dir,
        label_col=args.label_col,
        n_sample=args.n_sample,
    )
    test_dataset = BagsDataset_IMP(
        csv_file=args.test_csv,
        feature_dir=args.test_feature_dir,
        label_col=args.label_col,
        n_sample=args.n_sample,
    )

    # 评估阶段按原逻辑 batch_size=1
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -----------------------
    # Config for CV training
    # -----------------------
    config_params = {
        "dataset": train_dataset,
        "num_epochs": args.num_epochs,
        "device": args.device,
        "output_dir": args.output_dir,
        "n_splits": args.n_splits,
        "seed": args.seed,
    }

    # -----------------------
    # Hyperparameter search space (示例，可按需调整)
    # -----------------------
    hyper_params_grid = {
        "model_params": {
            "input_dim": [2048],
            "embed_dim_reduction": [256, 512, 1024],
            "embed_dim": [128, 256, 512],
            "Task": [2],          # 分类任务
            "output_dim": [3],    # 类别数（请根据你的数据集修改）
            "attention_dropout": [False],
            "normalize_attention": [True],
        },
        "optimizer_paras": {
            "learning_rate": [1e-3, 5e-4, 1e-4],
            "T_max": [5, 10, 15],
            "eta_min": [1e-6, 5e-6],
        },
        "early_stopping_params": {
            "patience": [15, 20],
            "verbose": [True],
            "delta": [0.0, 0.001, 0.005],
        },
        "criterion_params": {
            "label_smoothing": [0.0, 0.2, 0.4],
        },
    }

    # -----------------------
    # Optuna
    # -----------------------
    wrapped_objective = lambda trial: objective(
        trial=trial,
        hyper_params_grid=hyper_params_grid,
        config_params=config_params,
        output_dir=args.output_dir,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(wrapped_objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["best_trial_number"] = int(best_trial.number)

    with open(os.path.join(args.output_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print(f"[Optuna] Best trial: {best_trial.number}, Best AUC: {best_trial.value:.4f}")
    print(f"[Optuna] Best params saved to: {os.path.join(args.output_dir, 'best_params.json')}")

    # -----------------------
    # Evaluate best trial (fold checkpoints)
    # -----------------------
    best_trial_dir = os.path.join(args.output_dir, "trials", f"trial_{best_trial.number:04d}")

    # 注意：这里使用“model_params”的固定配置（与搜索空间一致）
    # 若你的 model_params 在 objective 里被 trial 改变，需要在 trial 输出目录读取 hyperparameters.json。
    with open(os.path.join(best_trial_dir, "hyperparameters.json"), "r", encoding="utf-8") as f:
        hp = json.load(f)
    model_params = hp["model_params"]

    test_aucs = evaluate_cross_validation_models(
        test_loader=test_loader,
        fold_dir=best_trial_dir,
        device=args.device,
        model_params=model_params,
        n_splits=args.n_splits,
    )
    plot_boxplot(test_aucs, os.path.join(args.output_dir, "test_auc_boxplot.pdf"), title="Test AUC Distribution")

    train_aucs = evaluate_cross_validation_models(
        test_loader=train_loader,
        fold_dir=best_trial_dir,
        device=args.device,
        model_params=model_params,
        n_splits=args.n_splits,
    )
    plot_boxplot(train_aucs, os.path.join(args.output_dir, "train_auc_boxplot.pdf"), title="Train AUC Distribution")

    with open(os.path.join(args.output_dir, "auc_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"train_aucs": train_aucs, "test_aucs": test_aucs}, f, ensure_ascii=False, indent=2)

    print("[Done] Evaluation finished.")


if __name__ == "__main__":
    main()
