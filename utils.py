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

    return images[:60000], one_hot_labels[:60000], images[60000:], one_hot_labels[60000:],
