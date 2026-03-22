import argparse
import os
import pickle
import random
from functools import partial

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from src.datasets.bags_dataset import BagsDatasetIMP
from src.optimization.hyperparameter_optimization import objective
from src.training.train_cv import evaluate_cross_validation_models


def set_seed(seed: int = 149) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MIL hyperparameter optimization.")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument("--feature-dir", type=str, required=True, help="Directory containing .pt feature files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for outputs and checkpoints.")
    parser.add_argument("--n-sample", type=int, default=30000, help="Maximum number of samples to load.")
    parser.add_argument("--num-epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-cpu", type=int, default=16, help="Number of CPU threads used by PyTorch.")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(149)
    torch.set_num_threads(args.num_cpu)
    print(f"Process ID: {os.getpid()}")

    train_dataset = BagsDatasetIMP(
        csv_file=args.train_csv,
        feature_dir=args.feature_dir,
        n_sample=args.n_sample,
    )
    test_dataset = BagsDatasetIMP(
        csv_file=args.test_csv,
        feature_dir=args.feature_dir,
        n_sample=args.n_sample,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    config_params = {
        "dataset": train_dataset,
        "num_epochs": args.num_epochs,
        "device": args.device,
        "output_dir": args.output_dir,
        "n_splits": args.n_splits,
    }

    hyper_params_grid = {
        "model_params": {
            "input_dim": [2048],
            "embed_dim": [256, 512],
            "task": [2],
            "output_dim": [3],
        },
        "optimizer_params": {
            "learning_rate": [1e-3, 5e-3, 1e-4],
            "T_max": [15, 25, 20],
            "eta_min": [1e-6, 5e-6],
        },
        "early_stopping_params": {
            "patience": [20, 25],
            "verbose": [True],
            "delta": [0.001, 0.005],
        },
        "criterion_params": {
            "label_smoothing": [0],
        },
    }

    wrapped_objective = partial(
        objective,
        hyper_params_grid=hyper_params_grid,
        config_params=config_params,
        output_dir=args.output_dir,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(wrapped_objective, n_trials=args.n_trials, n_jobs=1)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["best_trial_number"] = best_trial.number

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "best_params.pkl"), "wb") as fout:
        pickle.dump(best_params, fout)

    best_model_path = os.path.join(
        args.output_dir,
        "hyperparameter_search",
        f"trial_{best_trial.number}",
        "model.pickle",
    )
    with open(best_model_path, "rb") as fin:
        model = pickle.load(fin)

    torch.save(model.to("cpu"), os.path.join(args.output_dir, "best_model.pth"))

    trial_output_dir = os.path.join(
        args.output_dir,
        "hyperparameter_search",
        f"trial_{best_trial.number}",
    )

    test_aucs = evaluate_cross_validation_models(
        test_loader=test_dataloader,
        fold_dir=trial_output_dir,
        device=config_params["device"],
        num_folds=args.n_splits,
    )

    train_aucs = evaluate_cross_validation_models(
        test_loader=train_dataloader,
        fold_dir=trial_output_dir,
        device=config_params["device"],
        num_folds=args.n_splits,
    )

    auc_summary = {
        "train_aucs": train_aucs,
        "test_aucs": test_aucs,
    }

    with open(os.path.join(args.output_dir, "auc_summary.pkl"), "wb") as fout:
        pickle.dump(auc_summary, fout)

    print("Hyperparameter optimization finished.")
    print(f"Best trial: {best_trial.number}")
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
