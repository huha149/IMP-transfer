# Attention MIL Training Pipeline

A cleaned and anonymized PyTorch project for attention-based multiple instance learning (MIL)
with cross-validation and Optuna hyperparameter optimization.

## Project Structure

```text
github_ready_attention_mil/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_hyperparameter_search.py
└── src/
    ├── datasets/
    │   └── bags_dataset.py
    ├── models/
    │   └── attention_mil.py
    ├── optimization/
    │   └── hyperparameter_optimization.py
    ├── training/
    │   ├── early_stopping.py
    │   └── train_cv.py
    └── utils/
        └── save_loss_curve.py
```

## Notes

- Real file paths, private directory names, and environment-specific settings were removed.
- All comments and docstrings were rewritten in English.
- The code is organized for direct publication to GitHub, but you still need to provide your own dataset paths.

## Expected Data Format

### CSV files
The training and test CSV files should contain:
- column 0: `bag_id`
- column `slide_label`: class label

### Feature files
Each sample should have a corresponding PyTorch tensor file:
- `{feature_dir}/{bag_id}.pt`

## Installation

```bash
pip install -r requirements.txt
```

## Example Usage

```bash
python scripts/run_hyperparameter_search.py \
  --train-csv data/label_train.csv \
  --test-csv data/label_test.csv \
  --feature-dir data/features \
  --output-dir outputs/experiment_01
```

## Important Implementation Notes

- The current training pipeline assumes `batch_size=1`.
- Cross-validation uses `KFold`.
- Early stopping saves a full model checkpoint for each fold.
- Multi-class AUC is computed with `roc_auc_score(..., multi_class="ovr")`.

## Before Publishing

Review the following before pushing to GitHub:
1. Dataset paths
2. Output directories
3. Default hyperparameters
4. Any domain-specific file naming conventions
