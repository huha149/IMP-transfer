import os

import torch


class EarlyStopping:
    """
    Stop training when validation loss does not improve for a given number of epochs.
    """

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0.0,
        save_path: str = "",
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.save_path = os.path.join(save_path, "checkpoint.pt")

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.verbose:
            print(
                f"Validation loss improved ({self.val_loss_min:.6f} -> {val_loss:.6f}). "
                "Saving checkpoint."
            )
        torch.save(model, self.save_path)
        self.val_loss_min = val_loss
