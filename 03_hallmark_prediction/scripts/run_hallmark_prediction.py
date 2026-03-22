import argparse
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import random_split

from src.datasets.hallmark_bags_dataset import HallmarkBagsDataset
from src.training.train_cv import (
    evaluate_cv_best_models_on_test,
    train_with_cross_validation,
)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_bool(value: str) -> bool:
    """
    Parse a user string into boolean.
    """
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y", "shuffle"}:
        return True
    if value in {"false", "0", "no", "n", "no_shuffle"}:
        return False
    raise ValueError(f"Unsupported boolean-like value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an attention-based MIL model for hallmark score prediction."
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--hallmark", type=str, required=True, help="Target hallmark score column.")
    parser.add_argument("--device", type=str, required=True, help="Training device.")
    parser.add_argument(
        "--train-mode",
        type=str,
        required=True,
        choices=["scratch", "pretrained_finetune", "pretrained_freeze_att", "random_freeze_att"],
        help="Training strategy.",
    )
    parser.add_argument(
        "--shuffle",
        type=str,
        default="false",
        help="Whether to shuffle attention weights during training.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--train-csv", type=str, required=True, help="Training CSV file.")
    parser.add_argument("--test-csv", type=str, default=None, help="Optional explicit test CSV file.")
    parser.add_argument("--external-csv", type=str, default=None, help="Optional external evaluation CSV file.")
    parser.add_argument("--external-csv-old", type=str, default=None, help="Optional additional external CSV file.")
    parser.add_argument("--feature-dir", type=str, required=True, help="Feature directory for main dataset.")
    parser.add_argument("--external-feature-dir", type=str, default=None, help="Feature directory for external dataset.")
    parser.add_argument(
        "--external-feature-dir-old",
        type=str,
        default=None,
        help="Feature directory for the secondary external dataset.",
    )
    parser.add_argument(
        "--pretrained-imp-path",
        type=str,
        default=None,
        help="Path to the pretrained IMP model checkpoint used for attention initialization.",
    )
    parser.add_argument("--n-sample", type=int, default=1500, help="Maximum number of main samples to load.")
    parser.add_argument("--external-n-sample", type=int, default=3000, help="Maximum number of external samples.")
    parser.add_argument("--num-epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--num-cpu", type=int, default=4, help="Number of CPU threads.")
    return parser.parse_args()


def maybe_build_dataset(csv_path: str | None, feature_dir: str | None, hallmark: str, n_sample: int):
    """
    Build a dataset only when its CSV path is provided.
    """
    if not csv_path or not feature_dir:
        return None
    return HallmarkBagsDataset(
        csv_file=csv_path,
        feature_dir=feature_dir,
        hallmark=hallmark,
        n_sample=n_sample,
    )


def main():
    args = parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.num_cpu)
    print(f"Using hallmark: {args.hallmark}")
    print(f"Using device: {args.device}")
    print(f"Using train mode: {args.train_mode}")
    print(f"Using attention shuffle: {str_to_bool(args.shuffle)}")
    print(f"Process ID: {os.getpid()}")

    dataset = HallmarkBagsDataset(
        csv_file=args.train_csv,
        feature_dir=args.feature_dir,
        hallmark=args.hallmark,
        n_sample=args.n_sample,
    )

    if args.test_csv:
        test_dataset = HallmarkBagsDataset(
            csv_file=args.test_csv,
            feature_dir=args.feature_dir,
            hallmark=args.hallmark,
            n_sample=args.n_sample,
        )
        train_dataset = dataset
    else:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    external_feature_dir = args.external_feature_dir or args.feature_dir
    external_dataset = maybe_build_dataset(
        csv_path=args.external_csv,
        feature_dir=external_feature_dir,
        hallmark=args.hallmark,
        n_sample=args.external_n_sample,
    )
    external_dataset_old = maybe_build_dataset(
        csv_path=args.external_csv_old,
        feature_dir=args.external_feature_dir_old,
        hallmark=args.hallmark,
        n_sample=args.external_n_sample,
    )

    model_params = {
        "input_dim": 2048,
        "embed_dim": 512,
        "task": 1,
        "output_dim": 1,
        "dropout": False,
        "shuffle": str_to_bool(args.shuffle),
    }

    optimizer_params = {
        "learning_rate": 1e-4,
        "T_max": 25,
        "eta_min": 5e-6,
    }

    early_stopping_params = {
        "patience": 40,
        "verbose": True,
        "delta": 0.001,
    }

    config_params = {
        "seed": args.seed,
        "dataset": train_dataset,
        "num_epochs": args.num_epochs,
        "device": args.device,
        "output_dir": os.path.join(args.output_dir, "cross_validation"),
        "n_splits": args.n_splits,
        "train_mode": args.train_mode,
        "pretrained_path": args.pretrained_imp_path,
    }

    (
        best_rmse_model,
        best_mape_model,
        best_auc_model,
        fold_best_rmses,
        fold_best_mapes,
        fold_best_aucs,
    ) = train_with_cross_validation(
        optimizer_params=optimizer_params,
        early_stopping_params=early_stopping_params,
        model_params=model_params,
        config_params=config_params,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(best_rmse_model.to("cpu"), os.path.join(args.output_dir, "best_rmse_model.pth"))
    torch.save(best_mape_model.to("cpu"), os.path.join(args.output_dir, "best_mape_model.pth"))
    torch.save(best_auc_model.to("cpu"), os.path.join(args.output_dir, "best_auc_model.pth"))

    training_summary = {
        "model_params": model_params,
        "optimizer_params": optimizer_params,
        "early_stopping_params": early_stopping_params,
        "config_params": config_params,
        "fold_best_rmses": fold_best_rmses,
        "fold_best_mapes": fold_best_mapes,
        "fold_best_aucs": fold_best_aucs,
    }
    with open(os.path.join(args.output_dir, "training_summary.pickle"), "wb") as file:
        pickle.dump(training_summary, file)

    evaluate_cv_best_models_on_test(
        output_dir=args.output_dir,
        dataset=dataset,
        dataset_name="main_dataset",
        model_params=model_params,
        device=args.device,
        n_folds=args.n_splits,
        cv_subdir="cross_validation",
        save_attention_per_fold=True,
    )

    evaluate_cv_best_models_on_test(
        output_dir=args.output_dir,
        dataset=test_dataset,
        dataset_name="main_dataset_test",
        model_params=model_params,
        device=args.device,
        n_folds=args.n_splits,
        cv_subdir="cross_validation",
        save_attention_per_fold=True,
    )

    if external_dataset is not None:
        evaluate_cv_best_models_on_test(
            output_dir=args.output_dir,
            dataset=external_dataset,
            dataset_name="external_dataset",
            model_params=model_params,
            device=args.device,
            n_folds=args.n_splits,
            cv_subdir="cross_validation",
            save_attention_per_fold=True,
        )

    if external_dataset_old is not None:
        evaluate_cv_best_models_on_test(
            output_dir=args.output_dir,
            dataset=external_dataset_old,
            dataset_name="external_dataset_old",
            model_params=model_params,
            device=args.device,
            n_folds=args.n_splits,
            cv_subdir="cross_validation",
            save_attention_per_fold=True,
        )

    print("Hallmark prediction training and evaluation completed.")


if __name__ == "__main__":
    main()
