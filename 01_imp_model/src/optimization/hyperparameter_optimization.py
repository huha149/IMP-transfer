import os
import pickle

from src.training.train_cv import train_with_cross_validation


def objective(trial, hyper_params_grid: dict, config_params: dict, output_dir: str):
    """
    Optuna objective function for hyperparameter optimization.
    """
    model_params = {}
    optimizer_params = {}
    early_stopping_params = {}
    criterion_params = {}

    for key, values in hyper_params_grid["model_params"].items():
        model_params[key] = trial.suggest_categorical(key, values)

    for key, values in hyper_params_grid["optimizer_params"].items():
        optimizer_params[key] = trial.suggest_categorical(key, values)

    for key, values in hyper_params_grid["early_stopping_params"].items():
        early_stopping_params[key] = trial.suggest_categorical(key, values)

    for key, values in hyper_params_grid["criterion_params"].items():
        criterion_params[key] = trial.suggest_categorical(key, values)

    config_params_trial = dict(config_params)
    config_params_trial["output_dir"] = os.path.join(output_dir, "hyperparameter_search", f"trial_{trial.number}")

    best_model, avg_auc, fold_best_val_aucs = train_with_cross_validation(
        optimizer_params=optimizer_params,
        early_stopping_params=early_stopping_params,
        model_params=model_params,
        criterion_params=criterion_params,
        config_params=config_params_trial,
    )

    with open(os.path.join(config_params_trial["output_dir"], "model.pickle"), "wb") as fout:
        pickle.dump(best_model, fout)
        print(f"Trial {trial.number}: model saved.")

    with open(os.path.join(config_params_trial["output_dir"], "fold_auc_list.pickle"), "wb") as fout:
        pickle.dump(fold_best_val_aucs, fout)
        print(f"Trial {trial.number}: fold AUC list saved.")

    hyperparams = {
        "model_params": model_params,
        "optimizer_params": optimizer_params,
        "early_stopping_params": early_stopping_params,
        "criterion_params": criterion_params,
    }

    with open(os.path.join(config_params_trial["output_dir"], "hyperparameters.pickle"), "wb") as fout:
        pickle.dump(hyperparams, fout)
        print(f"Trial {trial.number}: hyperparameters saved.")

    return avg_auc
