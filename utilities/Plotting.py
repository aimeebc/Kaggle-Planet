"""Code to produce different plots to understand the fitting."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import LoadData
from sklearn.metrics import confusion_matrix


def plot_training_curves(folder):
    """Plot accuracy and loss curves saved to csv during training."""
    history = np.genfromtxt(folder+'/history.csv', delimiter=',', names=True)

    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['loss'],     label='training')
    ax.plot(history['epoch'], history['val_loss'], label='validation')
    ax.legend()
    ax.set(xlabel='epoch', ylabel='loss')
    fig.savefig(folder+'/loss.png')

    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['acc'],     label='training')
    ax.plot(history['epoch'], history['val_acc'], label='validation')
    ax.legend()
    ax.set(xlabel='epoch', ylabel='accuracy')
    fig.savefig(folder+'/accuracy.png')
    return


def plot_category_dist(preds, trues, categories, folder):
    """Show the distributions over categories."""
    pred_sums = np.sum(preds, axis=0)
    true_sums = np.sum(trues, axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(17), pred_sums, label='predictions')
    ax.plot(range(17), true_sums, label='truth')
    ax.legend()
    ax.set(xlabel='category', ylabel='sum')
    ax.set_xticks(range(17))
    ax.set_xticklabels(categories, minor=False, rotation=90)

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.35)

    fig.savefig(folder+'/categories.png')

    # And save a log version to show more detail on the rare classes
    plt.yscale('log')

    fig.savefig(folder+'/categories_log.png')

    return

if __name__ == '__main__':

    if len(sys.argv) == 2:
        folder = sys.argv[1]
    else:
        sys.exit('Please give the folder containing the results')

    KAGGLE_ROOT = os.path.abspath("input/")
    KAGGLE_LABEL_CSV = os.path.join(KAGGLE_ROOT, 'train_v2.csv')
    all_train_labels = LoadData.load_labels(KAGGLE_LABEL_CSV)
    unique_label_list = LoadData.get_unique_labels(all_train_labels)

    p_preds = np.load(folder + '/preds.npy')
    trues = np.load(folder + '/trues.npy')
    thresholds = np.load(folder + '/thresholds.npy')

    # Predictions now for classes
    preds = np.zeros_like(p_preds)
    for i in range(17):
        preds[:, i] = (p_preds[:, i] > thresholds[i]).astype(np.int)

    plot_training_curves(folder)
    plot_category_dist(preds, trues, unique_label_list, folder)
