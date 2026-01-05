from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class BagsDataset_IMP(Dataset):
    """Bag-level Dataset for MIL.

    该数据集假设：
    - 一个 CSV 文件提供每个 bag 的 ID 与标签
    - 每个 bag 的特征保存在 feature_dir 下的 "{bag_id}.pt" 文件中
      文件内容通常是 Tensor，形状 (num_instances, feature_dim)

    Parameters
    ----------
    csv_file:
        标签 CSV 路径，例如："path/to/labels/train.csv"
    feature_dir:
        特征目录，例如："path/to/features/"
    label_col:
        标签列名，默认 "slide_label"
    id_col:
        ID 列名，默认使用第 0 列（与原代码兼容）
    n_sample:
        可选：只加载前 n_sample 个样本，方便调试
    """

    def __init__(
        self,
        csv_file: str,
        feature_dir: str,
        label_col: str = "slide_label",
        id_col: Optional[Union[int, str]] = 0,
        n_sample: Optional[int] = None,
    ) -> None:
        self.bag_labels = pd.read_csv(csv_file)

        if n_sample is not None and n_sample < len(self.bag_labels):
            self.bag_labels = self.bag_labels.iloc[:n_sample].reset_index(drop=True)
            print(f"[Dataset] Loaded first {n_sample} samples.")
        else:
            print(f"[Dataset] Loaded {len(self.bag_labels)} samples.")

        self.feature_dir = feature_dir
        self.label_col = label_col
        self.id_col = id_col

    def __len__(self) -> int:
        return len(self.bag_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.bag_labels.iloc[idx]

        bag_id = row[self.id_col] if isinstance(self.id_col, str) else row.iloc[self.id_col]
        label = row[self.label_col]

        feature_path = os.path.join(self.feature_dir, f"{bag_id}.pt")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}. "
                f"Please check feature_dir and bag_id mapping."
            )

        bag_features = torch.load(feature_path, map_location="cpu")
        # 分类任务常用 long；回归可在训练阶段转换为 float
        bag_label = torch.tensor(label).long()

        return bag_features, bag_label
