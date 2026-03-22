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

## Recommended Publishing Strategy

This repository is structured for direct GitHub publication. The code was cleaned to remove:
- personal local paths
- machine-specific directories
- private workspace naming
- non-English comments
- environment-specific import hacks

## Suggested Usage Order

1. Train or tune the IMP model in `01_imp_model`
2. Export the checkpoint to be used by `02_gene_prediction`
3. Train the gene prediction model with or without pretrained IMP attention
4. Train the hallmark prediction model with or without pretrained IMP attention

## Example High-Level Workflow

### Step 1: Train the IMP model
```bash
python 01_imp_model/scripts/run_hyperparameter_search.py \
  --train-csv data/imp_train.csv \
  --test-csv data/imp_test.csv \
  --feature-dir data/features \
  --output-dir outputs/imp_experiment
```

### Step 2: Train the gene prediction model
```bash
python 02_gene_prediction/scripts/run_gene_prediction.py \
  --seed 149 \
  --gene TP53 \
  --device cuda:0 \
  --train-mode pretrained_freeze_att \
  --ce-weight true \
  --shuffle false \
  --train-csv data/gene_train.csv \
  --test-csv data/gene_test.csv \
  --feature-dir data/features \
  --output-dir outputs/gene_tp53 \
  --pretrained-imp-path outputs/imp_experiment/best_model.pth
```

### Step 3: Train the hallmark prediction model
```bash
python 03_hallmark_prediction/scripts/run_hallmark_prediction.py \
  --seed 149 \
  --hallmark "Angiogenesis score" \
  --device cuda:0 \
  --train-mode pretrained_freeze_att \
  --shuffle false \
  --train-csv data/label_gsva.csv \
  --feature-dir data/features \
  --output-dir outputs/angiogenesis_run \
  --pretrained-imp-path outputs/imp_experiment/best_model.pth
```

## Notes

- The three modules are intentionally separated so they can be maintained independently.
- Shared modeling ideas are preserved, but each module has its own training entry point.
- Review all dataset paths, output paths, and file naming conventions before publishing.
- Large model checkpoints, feature tensors, and CSV labels should normally not be committed to GitHub.

## License

Add a license file if you plan to make the repository public.
