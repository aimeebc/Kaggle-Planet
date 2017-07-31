"""Define all the callbacks that can be used."""

import keras
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def get_early_stopping_callback(epochs):
    """
    Early stopping to auto-stop training process.

    If model stops learning after n epochs, stop training.
    """
    cbck = keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=epochs,
                                         verbose=0,
                                         mode='auto')
    return cbck


def get_csv_callback(folder):
    """A callback that writes accuracies and losses to csv during training."""
    cbck = keras.callbacks.CSVLogger(folder+'/history.csv')
    return cbck


def get_best_weights_callback(folder):
    """Save the weights at their best during the training."""
    filepath = '/weights.best.hdf5'
    cbck = ModelCheckpoint(folder+filepath, monitor='val_acc',
                           verbose=1, save_best_only=True)
    return cbck


class LearningRateTracker(Callback):
    """A callback class which outputs the learning rate as it decays."""

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))
