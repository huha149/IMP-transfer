# Triple-Module Attention MIL Repository

This repository contains three related modules for attention-based multiple instance learning (MIL)
from pathology-derived bag features.

## Repository Structure

```text
github_triple_module_mil_repo/
├── README.md
├── .gitignore
├── requirements.txt
├── 01_imp_model/
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts/
│   └── src/
├── 02_gene_prediction/
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts/
│   └── src/
└── 03_hallmark_prediction/
    ├── README.md
    ├── requirements.txt
    ├── scripts/
    └── src/
```

## Module Overview

### 01_imp_model
The first module is the IMP prediction pipeline.

Main features:
- attention-based MIL model
- K-fold cross-validation
- Optuna-based hyperparameter search
- early stopping
- loss curve export

### 02_gene_prediction
The second module is the gene prediction pipeline and is designed to work together
with the IMP module.

Main features:
- attention-based MIL model for binary gene prediction
- pretrained attention initialization from the IMP module
- scratch / finetune / freeze-attention training modes
- weighted cross-entropy loss for imbalanced classification
- cross-validation evaluation
- external cohort evaluation
- attention export for downstream interpretation

### 03_hallmark_prediction
The third module is the hallmark prediction pipeline.

Main features:
- attention-based MIL regression for hallmark scores
- pretrained attention initialization from the IMP module
- scratch / finetune / freeze-attention training modes
- cross-validation evaluation
- external cohort evaluation
- attention export for downstream interpretation
- regression metrics including RMSE and Pearson correlation

