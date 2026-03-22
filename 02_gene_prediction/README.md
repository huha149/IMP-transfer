# Module 02 - Gene Prediction with Attention MIL

This folder is the **second module** of the repository and is designed to be placed
next to the previous IMP model module in the same GitHub project.

The code implements an attention-based MIL pipeline for **gene mutation prediction**
from bag-level pathology features, with support for:

- 5-fold cross-validation
- multiple training modes
- pretrained attention loading from the IMP model
- weighted cross-entropy loss for class-imbalanced binary classification
- external evaluation on multiple cohorts
- attention export for downstream interpretation

## Highlights

### Weighted Cross-Entropy Loss
This module explicitly supports **weighted cross-entropy loss** through the `CE_weight`
configuration flag. When enabled, class weights are computed from the number of positive
and negative samples inside the current split:

- negative class weight = `1 / num_negative`
- positive class weight = `1 / num_positive`

This is useful for gene prediction tasks where mutation labels are often imbalanced.

## Suggested Repository Layout

```text
repo/
├── 01_imp_model/
└── 02_gene_prediction/
    ├── README.md
    ├── requirements.txt
    ├── scripts/
    │   └── run_gene_prediction.py
    └── src/
        ├── datasets/
        │   └── gene_bags_dataset.py
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

### Mode description

- `scratch`: train the full model from random initialization
- `pretrained_finetune`: initialize the attention block from the IMP model and finetune all trainable layers
- `pretrained_freeze_att`: initialize the attention block from the IMP model and freeze it during training
- `random_freeze_att`: reinitialize the attention block randomly and freeze it

## Data Assumptions

### CSV file
The label CSV should contain:

- first column: bag identifier
- one binary label column for each gene, for example `TP53`, `KRAS`, `APC`, etc.

### Feature directory
Each bag must have a feature tensor stored as:

```text
{feature_dir}/{bag_id}.pt
```

## Example Usage

```bash
python scripts/run_gene_prediction.py \
  --seed 149 \
  --gene TP53 \
  --device cuda:0 \
  --train-mode pretrained_freeze_att \
  --ce-weight true \
  --shuffle false \
  --train-csv data/train_labels.csv \
  --test-csv data/test_labels.csv \
  --external-csv data/external_labels.csv \
  --feature-dir data/features \
  --external-feature-dir data/external_features \
  --output-dir outputs/tp53_run \
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
- cohort names
- output directory naming
- whether external evaluation datasets should be included in the public repository
