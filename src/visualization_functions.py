from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch

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
    
    if save_path and not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)

    labels_cpu = labels.cpu()
    normal_indices = np.where(labels_cpu == 0)[0]
    anomaly_indices = np.where(labels_cpu != 0)[0]

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


def draw_histogram_of_errors_by_class(errors, labels, save_path=None):
    """
    Tworzy i zapisuje histogram błędów rekonstrukcji, grupując je według klas.
    
    Args:
        errors (np.ndarray): Tablica błędów rekonstrukcji.
        labels (torch.Tensor): Tensor etykiet dla każdej próbki.
        save_path (str, optional): Ścieżka do zapisania wykresu.
    """
    if save_path and not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)

    labels_cpu = labels.cpu().numpy()
    unique_labels = sorted(np.unique(labels_cpu))
    
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.figure(figsize=(12, 7))
    
    for i, label_class in enumerate(unique_labels):
        indices = np.where(labels_cpu == label_class)[0]
        if len(indices) > 0:
            label_name = f"Anomaly Class {label_class}"
            if label_class == 0:
                label_name = "Normal (Class 0)"
            
            plt.hist(errors[indices], bins=50, alpha=0.7, color=colors(i), label=label_name)

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram of Reconstruction Errors by Class")
    plt.tight_layout()

    if save_path:
        plt.savefig(Path(save_path) / 'histogram_of_errors_by_class.png')


def draw_confusion_matrix(errors, targets, threshold=0, save_path=None, unsupervised=True):

    if unsupervised:
        y_true = targets.cpu().numpy()
        y_pred = np.where(errors > threshold, 1, 0)
        labels = ['normal', 'anomaly']
        num_classes = 2
    
    else:
        y_true = targets.cpu().numpy()
        y_pred = torch.argmax(errors, dim=1).cpu().numpy()
        num_classes = max(y_true.max(), y_pred.max()) + 1 
        labels = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

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

def visualize_autoencoder_reconstructions(autoencoder, dataset, save_path=None, num_samples=5):
    """
    Wizualizuje rekonstrukcje autoenkodera dla próbek normalnych i anomalnych.
    
    Args:
        autoencoder: Wytrenowany model autoenkodera
        dataset: Dataset testowy zawierający próbki z etykietami
        save_path: Ścieżka do zapisania wykresu
        num_samples: Liczba próbek każdego typu do wizualizacji (domyślnie 5)
    """
    import random
    
    if save_path and not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Znajdź indeksy próbek normalnych i anomalnych
    normal_indices = []
    anomaly_indices = []
    
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == 0:  # Normalne
            normal_indices.append(i)
        elif label == 1:  # Anomalie
            anomaly_indices.append(i)
    
    # Losowo wybierz próbki
    random.seed(42)  # Dla powtarzalności wyników
    selected_normal = random.sample(normal_indices, min(num_samples, len(normal_indices)))
    selected_anomaly = random.sample(anomaly_indices, min(num_samples, len(anomaly_indices)))
    
    # Przygotuj dane
    normal_originals = []
    normal_reconstructions = []
    anomaly_originals = []
    anomaly_reconstructions = []
    
    autoencoder.eval()
    with torch.no_grad():
        # Przetwórz próbki normalne
        for idx in selected_normal:
            img, _ = dataset[idx]
            img_batch = img.unsqueeze(0)  # Dodaj wymiar batch
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
            
            reconstruction = autoencoder(img_batch)
            
            normal_originals.append(img.cpu())
            normal_reconstructions.append(reconstruction.squeeze(0).cpu())
        
        # Przetwórz próbki anomalne
        for idx in selected_anomaly:
            img, _ = dataset[idx]
            img_batch = img.unsqueeze(0)  # Dodaj wymiar batch
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
            
            reconstruction = autoencoder(img_batch)
            
            anomaly_originals.append(img.cpu())
            anomaly_reconstructions.append(reconstruction.squeeze(0).cpu())
    
    # Tworzenie subplot
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 3, 12))
    
    # Pierwsza linia: oryginalne próbki normalne
    for i in range(num_samples):
        if i < len(normal_originals):
            img = normal_originals[i].squeeze()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Normal - Original {i+1}')
            axes[0, i].axis('off')
    
    # Druga linia: rekonstrukcje próbek normalnych
    for i in range(num_samples):
        if i < len(normal_reconstructions):
            img = normal_reconstructions[i].squeeze()
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f'Normal - Reconstruction {i+1}')
            axes[1, i].axis('off')
    
    # Trzecia linia: oryginalne próbki anomalne
    for i in range(num_samples):
        if i < len(anomaly_originals):
            img = anomaly_originals[i].squeeze()
            axes[2, i].imshow(img, cmap='gray')
            axes[2, i].set_title(f'Anomaly - Original {i+1}')
            axes[2, i].axis('off')
    
    # Czwarta linia: rekonstrukcje próbek anomalnych
    for i in range(num_samples):
        if i < len(anomaly_reconstructions):
            img = anomaly_reconstructions[i].squeeze()
            axes[3, i].imshow(img, cmap='gray')
            axes[3, i].set_title(f'Anomaly - Reconstruction {i+1}')
            axes[3, i].axis('off')
    
    plt.suptitle("Comparison of autoencoder's reconstruction", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(Path(save_path) / 'autoencoder_reconstructions.png', dpi=300, bbox_inches='tight')
