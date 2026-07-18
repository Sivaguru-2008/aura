import random
import numpy as np
import torch
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

def set_seed(seed=7):
    """Sets system-wide seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HistoryLogger:
    """Logs epoch metrics to a CSV file."""
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        
    def log(self, epoch, metrics):
        row = {"epoch": epoch}
        row.update(metrics)
        
        fieldnames = ["epoch"] + list(metrics.keys())
        file_exists = self.csv_path.is_file()
        
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

def plot_training_history(csv_path: Path, plots_dir: Path):
    """Generates training loss and validation AUROC plots from history.csv."""
    if not csv_path.is_file():
        return
        
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = []
    train_losses = []
    val_losses = []
    macro_aurocs = []
    
    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row.get("train_loss", 0.0)))
            val_losses.append(float(row.get("val_loss", 0.0)))
            macro_aurocs.append(float(row.get("macro_auroc", 0.5)))
            
    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", color="#1f77b4")
    plt.plot(epochs, val_losses, label="Val Loss", marker="x", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close()
    
    # Plot AUROC
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, macro_aurocs, label="Validation Macro-AUROC", marker="o", color="#2ca02c")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("Validation Macro-AUROC")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "auroc_curves.png", dpi=150)
    plt.close()
