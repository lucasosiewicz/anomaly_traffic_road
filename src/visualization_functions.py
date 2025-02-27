import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def draw_loss_curves(train_loss, val_loss, save_path=None):

    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Loss curves')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(Path(save_path) / 'loss_curves.png')


def draw_historgram_of_errors(errors, labels, save_path=None):
    
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)

    normal_indices = np.where(labels.cpu() == 0)[0]
    anomaly_indices = np.where(labels.cpu() == 1)[0]

    # Draw two histograms
    plt.figure(figsize=(10, 6))
    plt.hist(errors[normal_indices], bins=50, alpha=0.5, color='blue', label='Normal')
    plt.hist(errors[anomaly_indices], bins=50, alpha=0.5, color='red', label='Anomaly')
    plt.xlabel("Reconstruction error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title('Histogram of errors')
    plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path) / 'histogram_of_errors.png')