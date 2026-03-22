# Module 03 - Hallmark Prediction with Attention MIL

This folder is the **third module** of the repository and is intended to be placed
next to:

- `01_imp_model`
- `02_gene_prediction`

This module implements an attention-based MIL pipeline for **hallmark score prediction**
from bag-level pathology features.

## Main Features

- attention-based MIL regression
- 5-fold cross-validation
- multiple training modes
- pretrained attention initialization from the IMP model
- external cohort evaluation
- attention export for downstream interpretation
- regression metrics including RMSE and Pearson correlation
- optional AUC-style ranking view by binarizing targets with a median threshold

## Suggested Repository Layout

```text
repo/
├── 01_imp_model/
├── 02_gene_prediction/
└── 03_hallmark_prediction/
    ├── README.md
    ├── requirements.txt
    ├── scripts/
    │   └── run_hallmark_prediction.py
    └── src/
        ├── datasets/
        │   └── hallmark_bags_dataset.py
        ├── models/
        │   └── attention_mil.py
        └── training/
            ├── train_cv.py
            ├── early_stopping.py
            └── save_loss_curve.py
```

## Training Modes

Supported modes in this cleaned version:

- `scratch`
- `pretrained_finetune`
- `pretrained_freeze_att`
- `random_freeze_att`

## Data Assumptions

### CSV file
The label CSV should contain:

- first column: bag identifier
- one numeric hallmark score column for each target hallmark

### Feature directory
Each bag must have a feature tensor stored as:

```text
{feature_dir}/{bag_id}.pt
```

## Metrics

This module is primarily a **regression** pipeline.

Reported metrics include:

- RMSE
- Pearson correlation coefficient
- Pearson p-value
