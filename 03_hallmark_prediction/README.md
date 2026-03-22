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

The original code also derived an AUC-like metric by thresholding validation targets at the median.
That behavior is preserved for compatibility, but the main task remains regression.

## Example Usage

```bash
python scripts/run_hallmark_prediction.py \
  --seed 149 \
  --hallmark "Angiogenesis score" \
  --device cuda:0 \
  --train-mode pretrained_freeze_att \
  --shuffle false \
  --train-csv data/label_gsva.csv \
  --test-csv data/test_labels.csv \
  --external-csv data/cptac_labels.csv \
  --external-csv-old data/cptac_labels_old.csv \
  --feature-dir data/features \
  --external-feature-dir data/cptac_features \
  --external-feature-dir-old data/cptac_features_old \
  --output-dir outputs/angiogenesis_run \
  --pretrained-imp-path ../01_imp_model/best_model.pth
```

## Publishing Notes

This cleaned version removes:

- local absolute paths
- personal workspace names
- machine-specific directory structures
- Chinese comments and environment-specific import hacks

Before publishing, please review:

- default dataset paths
- hallmark naming conventions
- output directory naming
- whether external evaluation datasets should be included in the public repository
