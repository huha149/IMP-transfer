from __future__ import annotations

import os
from typing import List, Sequence, Optional

import matplotlib.pyplot as plt


def save_loss_curve(losses: Sequence[float], output_pdf_path: str, title: str = "Loss Curve") -> None:
    """Plot loss curve and save as PDF."""
    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), list(losses), marker="o", label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_pdf_path, format="pdf")
    plt.close()


def plot_boxplot(data: Sequence[float], output_path: str, title: str = "AUC Distribution") -> None:
    """Simple boxplot helper (替代原代码中缺失的 boxplot.py)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.boxplot(list(data), vert=True, showmeans=True)
    plt.ylabel("AUC")
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path, format="pdf")
    plt.close()
