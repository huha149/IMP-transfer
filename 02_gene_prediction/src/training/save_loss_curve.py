import matplotlib.pyplot as plt


def save_loss_curve(losses, output_pdf_path: str) -> None:
    """
    Plot and save a loss curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(output_pdf_path, format="pdf")
    plt.close()
    print(f"Loss curve saved to {output_pdf_path}")
