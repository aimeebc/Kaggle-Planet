"""Helper functions."""

import numpy as np


def get_prediction_labels(p_preds, thresholds, label_list):
    """Take probability predictions, if they pass thresholds, save labels."""
    all_labels = []

    preds = np.zeros_like(p_preds)

    for i, pred in enumerate(p_preds):
        labels = ''
        for j in range(17):
            if pred[j] > thresholds[j]:
                preds[i, j] = 1
                labels += label_list[j] + ' '

        all_labels.append(labels)
    return all_labels
