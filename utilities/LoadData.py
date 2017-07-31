"""Code to load and preprocess the data."""

from __future__ import division
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg


def load_jpgs(folder, filenames, limit=None, channels=4):
    """Load jpgs in filenames up to limit, return np array with n channels."""
    n_files = limit if limit is not None else len(filenames)
    jpg_sample = np.empty((n_files, 256, 256, 4), dtype='uint8')

    for file_num, filename in enumerate(filenames[:n_files]):
        jpg_sample[file_num] = mpimg.imread(os.path.join(folder, filename))[:, :, :]

    # Restrict channels if necessary
    if channels > 0 and channels < 4:
        jpg_sample = jpg_sample[:, :, :, :channels]

    return jpg_sample


def load_labels(filename, limit=None):
    """Read lists of labels from csv to a panda dataframe."""
    all_labels = pd.read_csv(filename)

    if limit is not None and limit < len(all_labels) and limit > 0:
        all_labels = all_labels[:limit]

    return all_labels


def get_unique_labels(dataframe):
    """Take panda dataframe, return list of unique tags."""
    label_list = []

    for tag_str in dataframe.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)

    return label_list


def do_one_hot_encoding(dataframe, label_list):
    """Add onehot features for every unique label."""
    for label in label_list:
        dataframe[label] = dataframe['tags'] \
                           .apply(lambda x: 1 if label in x.split(' ') else 0)

    return dataframe


def batch_generator(list_of_image_names, batch_size,
                    channels=4, all_labels=None):
    """
    Generator for training or validation images with labels.

    A function to load the images from a list of filenames and yield them
    in batches of batch_size along with the corresponding labels.
    """
    n_images = len(list_of_image_names)

    start_index = 0

    while 1:

        end_index = start_index + batch_size
        images = np.empty((batch_size, 256, 256, 4), dtype='uint8')

        if end_index <= n_images:
            for file_num, filename in enumerate(list_of_image_names[start_index:end_index]):
                images[file_num] = mpimg.imread(filename)[:, :, :]

            if all_labels is not None:
                labels = all_labels[start_index:end_index, :]

        else:

            done = n_images - start_index
            to_do = end_index - n_images
            end_index = n_images

            for file_num, filename in enumerate(list_of_image_names[start_index:end_index]):
                images[file_num] = mpimg.imread(filename)[:, :, :]

            if all_labels is not None:
                labels = all_labels[start_index:end_index, :]

            start_index = 0
            end_index = to_do

            for file_num, filename in enumerate(list_of_image_names[start_index:end_index]):
                images[done + file_num] = mpimg.imread(filename)[:, :, :]

            if all_labels is not None:
                labels = np.concatenate((labels, all_labels[start_index:end_index, :]))

        # Restrict channels if necessary
        if channels > 0 and channels < 4:
            images = images[:, :, :, :channels]

        start_index = end_index

        if all_labels is not None:
            yield images, labels
        else:
            yield images
