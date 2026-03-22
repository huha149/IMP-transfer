import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class HallmarkBagsDataset(Dataset):
    """
    Dataset for hallmark score prediction from bag-level feature tensors.

    Args:
        csv_file: CSV file containing bag identifiers and hallmark scores.
        feature_dir: Directory containing `.pt` feature files.
        hallmark: Name of the target hallmark column.
        n_sample: Optional number of rows to load.
    """

    def __init__(self, csv_file: str, feature_dir: str, hallmark: str, n_sample: int | None = None):
        self.bag_labels = pd.read_csv(csv_file)

        if n_sample is not None and n_sample < len(self.bag_labels):
            self.bag_labels = self.bag_labels.iloc[:n_sample]
            print(f"Loaded the first {n_sample} samples.")
        else:
            print("Loaded all available samples.")

        self.feature_dir = feature_dir
        self.hallmark = hallmark

    def __len__(self) -> int:
        return len(self.bag_labels)

    def __getitem__(self, idx: int):
        bag_id = self.bag_labels.iloc[idx, 0]
        bag_label = torch.tensor(self.bag_labels.loc[idx, self.hallmark], dtype=torch.float32)

        bag_feature_path = os.path.join(self.feature_dir, f"{bag_id}.pt")
        bag_features = torch.load(bag_feature_path).float()

        return bag_features, bag_label.float(), str(bag_id)
