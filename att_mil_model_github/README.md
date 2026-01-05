# att_mil_model (GitHub-ready)

本仓库是对你提供的 MIL 代码进行整理后的“可公开到 GitHub”的版本：

- ✅ 模型名称统一为 `att_mil_model`
- ✅ 删除所有硬编码真实路径，统一使用占位路径（`path/to/...`），通过命令行参数传入
- ✅ 采用更清晰的包结构与相对导入（`src/` layout）
- ✅ 增加更规范的 docstring / 注释
- ✅ 修复训练中常见错误：`CrossEntropyLoss` 不应对 logits 先做 softmax（AUC 计算仍使用 softmax 概率）

## 目录结构

```text
.
├─ src/att_mil_model/
│  ├─ models/att_mil_model.py          # 模型（AttentionModule + att_mil_model）
│  ├─ data/bags_dataset.py             # 数据集
│  ├─ training/train_cv.py             # 交叉验证训练 & 评估
│  ├─ training/early_stopping.py       # 早停
│  ├─ tuning/hyper_params_optimizer.py # Optuna objective
│  └─ utils/plotting.py                # loss 曲线 & boxplot
├─ scripts/main_hyper_params.py        # 入口脚本（Optuna + 评估）
├─ configs/example_paths.yaml          # 占位配置
├─ requirements.txt
└─ .gitignore
```

## 数据要求

- CSV：至少包含一列 bag_id（默认第 0 列）和一列标签 `slide_label`（可通过参数修改）
- 特征文件：`feature_dir/{bag_id}.pt`，内容为 Tensor（例如形状 `(num_instances, feature_dim)`）

## 安装

```bash
pip install -r requirements.txt
```

推荐以可编辑模式安装（便于脚本 import）：

```bash
pip install -e .
```

## 运行（示例）

```bash
python scripts/main_hyper_params.py \
  --train_csv path/to/labels/train.csv \
  --train_feature_dir path/to/features/train \
  --test_csv path/to/labels/test.csv \
  --test_feature_dir path/to/features/test \
  --output_dir path/to/output \
  --device cuda:0
```

> 注意：请根据你的实际任务修改搜索空间中的 `output_dim`（类别数）以及 `input_dim`（特征维度）。
