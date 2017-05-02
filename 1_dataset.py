# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy import io


def load_dataset(filename):
    data_dict = io.loadmat(filename)
    images = data_dict["X"]
    images = np.transpose(images, (3, 0, 1, 2))
    labels = data_dict["y"]
    return images, labels


if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    
    # (N, 32, 32, 3), (N, 1)
    images, labels = load_dataset(FILENAME)
    print(images.shape, labels.shape)

