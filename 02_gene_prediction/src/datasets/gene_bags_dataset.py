import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class GeneBagsDataset(Dataset):
    """
    Dataset for gene mutation prediction from bag-level feature tensors.

    Args:
        csv_file: CSV file containing bag identifiers and gene labels.
        feature_dir: Directory containing `.pt` feature files.
        gene: Name of the target gene column.
        n_sample: Optional number of rows to load.
    """

    def __init__(self, csv_file: str, feature_dir: str, gene: str, n_sample: int | None = None):
        self.bag_labels = pd.read_csv(csv_file)

        if n_sample is not None and n_sample < len(self.bag_labels):
            self.bag_labels = self.bag_labels.iloc[:n_sample]
            print(f"Loaded the first {n_sample} samples.")
        else:
            print("Loaded all available samples.")

        self.feature_dir = feature_dir
        self.gene = gene

    def __len__(self) -> int:
        return len(self.bag_labels)

    def __getitem__(self, idx: int):
        bag_id = self.bag_labels.iloc[idx, 0]
        bag_label = int(self.bag_labels.loc[idx, self.gene])

        bag_feature_path = os.path.join(self.feature_dir, f"{bag_id}.pt")
        bag_features = torch.load(bag_feature_path).float()

        return bag_features, bag_label, str(bag_id)
