"""Utility functions for optimisations."""

from __future__ import division
import numpy as np
from sklearn.metrics import fbeta_score


def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    """
    Calculate the f2 scores to select the best thresholds.

    The predictions are given as probabilities between 0 & 1,
    they must be turned into a binary label for whether the image
    has the class or not. To do that a threshold is necessary.
    Rather than a flat threshold of say 0.2, a popular value for this
    competition, this approach indepently optimises the threshold performance
    per label to give the best f2 score. This approach was described by anokas
    https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
    on the forums and this is his function.
    """
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2]*17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x
