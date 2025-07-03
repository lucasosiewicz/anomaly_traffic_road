import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time

def count_best_threshold(reconstruction_error, targets, unsupervised=True):
    if unsupervised:
        best_threshold = 0
        best_accuracy = 0    
        best_precision = 0
        best_recall = 0
        best_f1 = 0

        targets = targets.cpu()
        # Convert targets to binary: 0 (normal) vs 1 (anomaly)
        binary_targets = torch.where(targets > 0, 1, 0)

        for threshold in np.linspace(min(reconstruction_error), max(reconstruction_error), 100):
            predictions = torch.where(reconstruction_error > threshold, 1, 0)

            accuracy = accuracy_score(binary_targets, predictions)
            precision = precision_score(binary_targets, predictions)
            recall = recall_score(binary_targets, predictions)
            f1 = f1_score(binary_targets, predictions)

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
        precision = precision_score(targets, reconstruction_error, average='macro')
        recall = recall_score(targets, reconstruction_error, average='macro')
        f1 = f1_score(targets, reconstruction_error, average='macro')

        return 0, accuracy, precision, recall, f1

def measure_single_sample_inference_time(model, dataloader, device, is_unsupervised, is_sequence):
    if dataloader is None or len(dataloader) == 0:
        return None

    try:
        test_iter = iter(dataloader)
        sample_batch = next(test_iter)
    except StopIteration:
        return None

    if isinstance(sample_batch, (list, tuple)):
        if len(sample_batch[0]) == 0:
             return None
        if is_sequence:
            single_sample = sample_batch[0][0].unsqueeze(0)
        else:
            single_sample = sample_batch[0][0].unsqueeze(0)
    else:
        if len(sample_batch) == 0:
            return None
        single_sample = sample_batch[0].unsqueeze(0)

    model = model.to(device)
    single_sample = single_sample.to(device)
    model.eval()
    with torch.no_grad():
        start_inference_time = time.time()
        if is_unsupervised:
            _ = model(single_sample)
        else:
            _ = model(single_sample)
        end_inference_time = time.time()
    
    single_sample_inference_duration_ms = (end_inference_time - start_inference_time) * 1000
    return single_sample_inference_duration_ms