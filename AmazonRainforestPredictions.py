"""
Amazon rainforest convolutional neural network on satellite data.

This program loads the best weights, thresholds and predictions from
trained models to calculate the cross-fold validation and generates
predictions and a submission file for the test data.
"""

from __future__ import print_function
import os

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

import keras
import tensorflow as tf
import time

from models import Inception3
from utilities import LoadData
from utilities import Helpers


# ----------------------------------------------------------
# Set job info

n_folds = 10
folder = 'Results/TestInception10fold'

GPU = False

# ----------------------------------------------------------

# ----------------------------------------------------------
# Set model parameters
# ----------------------------------------------------------

learning_rate = 1E-3
n_epochs = 10  # 10
decay_rate = learning_rate / n_epochs

# ----------------------------------------------------------
# Session setup
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
# Load model and compile
# ----------------------------------------------------------

base_model, model = Inception3.get_models()

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate, decay=decay_rate),
              metrics=['accuracy'])


all_f2_scores = []

for fold in range(n_folds):

    fold_folder = folder + '/fold_' + str(fold)

    # ----------------------------------------------------------
    # Load weights, thresholds, predictions & truth data
    # ----------------------------------------------------------

    best_weights_file = fold_folder + '/weights.best.hdf5'
    preds_file = fold_folder + '/preds.npy'
    trues_file = fold_folder + '/trues.npy'
    thresholds_file = fold_folder + '/thresholds.npy'

    model.load_weights(best_weights_file, by_name=False)

    thresholds = np.load(thresholds_file)

    preds = np.load(preds_file)
    trues = np.load(trues_file)

    # ----------------------------------------------------------
    # Calculate f2 scores
    # ----------------------------------------------------------

    # Predictions now for classes
    full_preds = np.zeros_like(preds)
    for i in range(17):
        full_preds[:, i] = (preds[:, i] >
                            thresholds[i]).astype(np.int)

    score = fbeta_score(trues, full_preds, beta=2, average='samples')
    print("New f2 is : ", score)

    all_f2_scores.append(score)

# ----------------------------------------------------------
# Calculate f2 scores after cross-fold validation
# ----------------------------------------------------------

all_f2_scores = np.array(all_f2_scores)
average_f2_score = np.mean(all_f2_scores)
print("K-fold validation f2 score is ", average_f2_score)

# ----------------------------------------------------------
# Select best model, load weights and thresholds
# ----------------------------------------------------------

# Get index for best performing fold
max_index = np.argmax(all_f2_scores)
print("Best fold selected with index ", max_index)
print("This fold gave the f2_score of ", all_f2_scores[max_index])

fold_folder = folder + '/fold_' + str(max_index)

best_weights_file = fold_folder + '/weights.best.hdf5'
thresholds_file = fold_folder + '/thresholds.npy'

model.load_weights(best_weights_file, by_name=False)
thresholds = np.load(thresholds_file)

# ----------------------------------------------------------
# Load test data
# ----------------------------------------------------------

KAGGLE_ROOT = os.path.abspath("input/")
KAGGLE_LABEL_CSV = os.path.join(KAGGLE_ROOT, 'train_v2.csv')
KAGGLE_JPEG_TEST_1 = os.path.join(KAGGLE_ROOT, 'test-jpg')
KAGGLE_JPEG_TEST_2 = os.path.join(KAGGLE_ROOT, 'test-jpg-additional')
KAGGLE_SUBMISSION_IN = os.path.join(KAGGLE_ROOT, 'sample_submission_v2.csv')

print("Loading labels")
train_labels = LoadData.load_labels(KAGGLE_LABEL_CSV)
unique_label_list = LoadData.get_unique_labels(train_labels)

# Creating list of test filenames
test_data = LoadData.load_labels(KAGGLE_SUBMISSION_IN)
test_jpg_list = test_data['image_name'].tolist()
full_test_jpg_list = []
for jpg_name in test_jpg_list[:]:
    if 'test' in jpg_name:
        full_test_jpg_list.append(os.path.join(KAGGLE_JPEG_TEST_1,
                                               jpg_name + '.jpg'))
    elif 'file' in jpg_name:
        full_test_jpg_list.append(os.path.join(KAGGLE_JPEG_TEST_2,
                                               jpg_name + '.jpg'))

# ----------------------------------------------------------
# Create test data generator
# ----------------------------------------------------------

batch_size = 64

n_test_events = len(full_test_jpg_list)

test_steps_per_epoch = np.floor(float(n_test_events) / batch_size)

test_generator = LoadData.batch_generator(full_test_jpg_list,
                                          batch_size,
                                          channels=3)

# ----------------------------------------------------------
# Make predictions on test data
# ----------------------------------------------------------

print("Making predictions for test images")
# First predictions from the model as probabilities
full_p_test_preds = model.predict_generator(test_generator,
                                            test_steps_per_epoch+1)

# Predictions now for classes
prediction_labels = Helpers.get_prediction_labels(full_p_test_preds[:n_test_events],
                                                  thresholds,
                                                  unique_label_list)

# ----------------------------------------------------------
# Create submission filenames
# ----------------------------------------------------------

# Save preds to file
np.save(folder + '/test_preds.npy', full_p_test_preds, allow_pickle=False)

results_file = "/final_submission_results.csv"
print('Writing submission file to ', results_file)

final_data = [[os.path.basename(filename).split(".")[0], tags]
              for filename, tags in zip(full_test_jpg_list,
                                        prediction_labels)]
final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
final_df.to_csv(folder + results_file, index=False)

print("Running time")
print("--- %s seconds ---" % (time.time() - start_time))
