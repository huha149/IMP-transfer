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
