# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import src.utils as utils


if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    # (N, 32, 32, 3), (N, 1)
    train, val = utils.load_dataset(FILENAME)
    print(train.images.shape, train.labels.shape, val.images.shape, val.labels.shape)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(train.images[0])
    print(train.labels[0])
    plt.show()
