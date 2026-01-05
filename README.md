
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

## IMP-classifier模型最佳参数


```bash
model_params = {'input_dim': 2048,  #输入特征维度
                'embed_dim_reduction':512,   #特征降维特征维度
                'embed_dim': 256,    #输隐藏特征维度
                'Task': 2, #任务类别，Regression:1或者Classification:2
                'output_dim': 3 #3个病变类别    
                }

# 加载预训练模型
model = torch.load("IMP_classifier.pth")

```
