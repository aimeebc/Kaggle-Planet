"""
Amazon rainforest convolutional neural network on satellite data.

A program to train a convolutional neural network on the
Amazon Rainforest satellite data as part of a kaggle competition.
This one uses a generator.
The model used is the Inception v3 model provided with Keras and the
manner in which it is fine tuned here follows the instructions on the
Keras applications website: https://keras.io/applications/#inceptionv3
"""

from __future__ import print_function
import os
import sys

import numpy as np
from sklearn.metrics import fbeta_score

import keras
import tensorflow as tf
import time

from models import CallbackDefs
from models import Inception3
from utilities import LoadData
from utilities import Optimisations as Opt

# ----------------------------------------------------------
# Set run details
# ----------------------------------------------------------
JOB_ID = '002'

if len(sys.argv) > 1:
    fold_no = int(sys.argv[1])
else:
    fold_no = 0
print("Fold number ", fold_no)
GPU = False

# For GPU jobs
if GPU:
    try:
        JOB_ID = os.environ['JOB_ID']
    except:
        sys.exit('Error: Set the JOB_ID"')


# Define where results will be stored
folder = 'train-forest-Inception-%s' % JOB_ID
while os.path.exists(folder):
    JOB_ID = JOB_ID + '_1'
    folder = 'train-forest-Inception-%s' % JOB_ID
folder = folder + '/fold_' + str(fold_no)
os.makedirs(folder)
print("Writing results to ", folder)

# ----------------------------------------------------------
# Load data, select what to use
# ----------------------------------------------------------

KAGGLE_ROOT = os.path.abspath("input/")
KAGGLE_JPEG_DIR = os.path.join(KAGGLE_ROOT, 'train-jpg')
KAGGLE_LABEL_CSV = os.path.join(KAGGLE_ROOT, 'train_v2.csv')

print("Loading labels")
all_train_labels = LoadData.load_labels(KAGGLE_LABEL_CSV)
unique_label_list = LoadData.get_unique_labels(all_train_labels)
all_train_labels = LoadData.do_one_hot_encoding(all_train_labels,
                                                unique_label_list)

print("Creating list of filenames")
jpg_list = all_train_labels['image_name'].tolist()
jpg_list = [os.path.join(KAGGLE_JPEG_DIR, jpg_name + '.jpg')
            for jpg_name in jpg_list]

# Define the training and validation sets
# Full set = 40479
n_total = len(jpg_list)


# The k-fold validation is done manually in this way so that different folds
# can be run concurrently, the results are later combined using
# AmazonRainforestPredictions.
valid_start = fold_no * 4048
valid_end = (fold_no + 1) * 4048
if valid_end > n_total:
    valid_end = n_total

print("Fold ", fold_no, " from ", valid_start, " to ", valid_end)

valid_jpg_list = jpg_list[valid_start:valid_end]
train_jpg_list = jpg_list
del train_jpg_list[valid_start:valid_end]

# Remove filenames to just keep binary data
full_set_labels = all_train_labels.values[:, 2:]

valid_labels = full_set_labels[valid_start:valid_end, :]
train_labels = full_set_labels
train_labels = np.delete(train_labels,
                         np.arange(valid_start, valid_end, 1), axis=0)

print("Size of training sample ", len(train_jpg_list))
print("Size of validation sample ", len(valid_jpg_list))

print("Size of training labels ", len(train_labels))
print("Size of validation labels ", len(valid_labels))

# ----------------------------------------------------------
# Set hyperparameters
# ----------------------------------------------------------

batch_size = 64

learning_rate = 1E-3
n_epochs = 10  # 10
decay_rate = learning_rate / n_epochs

callbacks_list = [CallbackDefs.get_csv_callback(folder),
                  CallbackDefs.get_early_stopping_callback(3),
                  CallbackDefs.get_best_weights_callback(folder),
                  CallbackDefs.LearningRateTracker()]

n_train_events = len(train_jpg_list)
n_valid_events = len(valid_jpg_list)

train_steps_per_epoch = np.floor(float(n_train_events) / batch_size)
valid_steps_per_epoch = np.floor(float(n_valid_events) / batch_size)

# ----------------------------------------------------------
# Second fit hyperparameters
# ----------------------------------------------------------

learning_rate_2 = 1E-5
n_epochs_2 = 30  # 30
decay_rate_2 = learning_rate_2 / n_epochs_2

# ----------------------------------------------------------
# Create generators
# ----------------------------------------------------------

# Training
train_generator = LoadData.batch_generator(train_jpg_list,
                                           batch_size,
                                           channels=3,
                                           all_labels=train_labels)

# Validation
valid_generator = LoadData.batch_generator(valid_jpg_list,
                                           batch_size,
                                           channels=3,
                                           all_labels=valid_labels)
# Validation for predictions
# Use a fresh validator so that it starts from the beginning of the sample
# and it is easier to associate the correct labels
predict_valid_generator = LoadData.batch_generator(valid_jpg_list,
                                                   batch_size,
                                                   channels=3,
                                                   all_labels=valid_labels)

# ----------------------------------------------------------
# Start session
# ----------------------------------------------------------

print("Starting session")

start_time = time.time()

if GPU:
    # GPU setup
    session_conf = tf.ConfigProto()
else:
    # Limit available CPU to one thread
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)

keras.backend.tensorflow_backend.set_session(tf.Session(config=session_conf))

# ----------------------------------------------------------
# Load and compile model
# ----------------------------------------------------------

base_model, model = Inception3.get_models()

print(model.summary())

# First: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate, decay=decay_rate),
              metrics=['accuracy'])

# ----------------------------------------------------------
# Fit model
# ----------------------------------------------------------

print("Fitting model...")

# Train the model on the new data for a few epochs
model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=n_epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=valid_generator,
                    validation_steps=valid_steps_per_epoch)

# At this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# Let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# We chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# ----------------------------------------------------------
# Recompile and refit model
# ----------------------------------------------------------

# We need to recompile the model for these modifications to take effect
# We use a low learning rate
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate_2,
                                              decay=decay_rate_2),
              metrics=['accuracy'])

# We train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers)
model.fit_generator(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=n_epochs_2,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=valid_generator,
                    validation_steps=valid_steps_per_epoch)

model.save(folder + '/model.h5')  # save final trained model
print("Running time")
print("--- %s seconds ---" % (time.time() - start_time))

# ----------------------------------------------------------
# Load best weights to make predictions on validation data
# ----------------------------------------------------------

weights_filepath = folder + '/weights.best.hdf5'
model.load_weights(weights_filepath, by_name=False)

# ----------------------------------------------------------
# Make predictions on validation data
# ----------------------------------------------------------

print("Making predictions")

# First predictions from the model as probabilities
full_p_preds = model.predict_generator(predict_valid_generator,
                                       valid_steps_per_epoch+1)
# Truth must be same type to calculate score
full_trues = np.array(valid_labels, np.uint8)

# Need to slice predictions because generator produces too many, wraps around
full_p_preds = full_p_preds[0:n_valid_events]

# Choose the thresholds
full_thresholds = Opt.optimise_f2_thresholds(full_trues, full_p_preds)
print("Set of thresholds chosen: ", full_thresholds)

# Predictions now for classes
full_preds_var = np.zeros_like(full_p_preds)
for i in range(17):
    full_preds_var[:, i] = (full_p_preds[:, i] >
                            full_thresholds[i]).astype(np.int)

# ----------------------------------------------------------
# Evaluate model
# ----------------------------------------------------------

print("Model performance (loss, accuracy)")
print("Train: %.4f, %.4f" % tuple(model.evaluate_generator(train_generator,
                                                           train_steps_per_epoch)))
print("Valid: %.4f, %.4f" % tuple(model.evaluate_generator(valid_generator,
                                                           valid_steps_per_epoch)))

score = fbeta_score(full_trues, full_preds_var, beta=2, average='samples')
print("New f2 is : ", score)

# ----------------------------------------------------------
# Save all data
# ----------------------------------------------------------
np.save(folder + '/preds.npy', full_p_preds, allow_pickle=False)

np.save(folder + '/trues.npy', full_trues, allow_pickle=False)

full_thresholds_np = np.array(full_thresholds)
np.save(folder + '/thresholds.npy', full_thresholds_np, allow_pickle=False)
