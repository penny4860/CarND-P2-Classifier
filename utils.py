# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import io


def load_dataset(filename):
    data_dict = io.loadmat(filename)
    images = data_dict["X"]
    images = np.transpose(images, (3, 0, 1, 2))
    labels = data_dict["y"].reshape(-1,)
    labels[labels==10] = 0
    
    one_hot_labels = np.zeros((len(labels), 10))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    one_hot_labels = one_hot_labels.reshape(-1, 10)
    
    train_set = Dataset(images[:60000], one_hot_labels[:60000])
    validation_set = Dataset(images[60000:], one_hot_labels[60000:])

    return train_set, validation_set


class Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        if self.epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self.images = self.images[perm0]
            self.labels = self.labels[perm0]

        if start + batch_size > self._num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self.images[start:self._num_examples]
            labels_rest_part = self.labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.images = self.images[perm]
                self.labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.images[start:end], self.labels[start:end]
