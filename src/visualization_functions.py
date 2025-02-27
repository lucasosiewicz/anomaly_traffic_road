from sklearn.metrics import confusion_matrix
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


def draw_confusion_matrix(errors, targets, threshold, save_path=None):

    y_true = targets.cpu().numpy()
    y_pred = np.where(errors > threshold, 1, 0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['normal', 'anomaly'], rotation=45)
    plt.yticks(tick_marks, ['normal', 'anomaly'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_path:
        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_path) / 'confusion_matrix.png')