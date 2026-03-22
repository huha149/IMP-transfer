import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class BagsDatasetIMP(Dataset):
    """
    Dataset for loading bag-level features and labels.

    Args:
        csv_file: Path to the CSV file containing sample metadata.
        feature_dir: Directory containing bag feature `.pt` files.
        n_sample: Optional number of samples to load.
    """

    def __init__(self, csv_file: str, feature_dir: str, n_sample: int | None = None):
        self.bag_labels = pd.read_csv(csv_file)

        if n_sample is not None and n_sample < len(self.bag_labels):
            self.bag_labels = self.bag_labels.iloc[:n_sample]
            print(f"Loaded {n_sample} samples.")
        else:
            print("Loaded all samples.")

        self.feature_dir = feature_dir

    def __len__(self) -> int:
        return len(self.bag_labels)

    def __getitem__(self, idx: int):
        bag_id = self.bag_labels.iloc[idx, 0]
        bag_label = torch.tensor(self.bag_labels.loc[idx, "slide_label"])

        bag_feature_path = os.path.join(self.feature_dir, f"{bag_id}.pt")
        bag_features = torch.load(bag_feature_path).float()

        return bag_features, bag_label.long()
