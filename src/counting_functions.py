import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def count_best_threshold(reconstruction_error, targets, unsupervised=True):

    if unsupervised:
        best_threshold = 0
        best_accuracy = 0    
        best_precision = 0
        best_recall = 0
        best_f1 = 0

        targets = targets.cpu()

        for threshold in np.linspace(min(reconstruction_error), max(reconstruction_error), 100):
            predictions = torch.where(reconstruction_error > threshold, 1, 0)

            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions)
            recall = recall_score(targets, predictions)
            f1 = f1_score(targets, predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_accuracy, best_precision, best_recall, best_f1
    
    else:
        reconstruction_error = torch.argmax(reconstruction_error, axis=1).cpu()
        accuracy = accuracy_score(targets, reconstruction_error)
        precision = precision_score(targets, reconstruction_error)
        recall = recall_score(targets, reconstruction_error)
        f1 = f1_score(targets, reconstruction_error)

        return accuracy, precision, recall, f1