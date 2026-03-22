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

