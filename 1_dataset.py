# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import utils


if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    # (N, 32, 32, 3), (N, 1)
    images, labels, images_val, labels_val = utils.load_dataset(FILENAME)
    print(images.shape, labels.shape, images_val.shape, labels_val.shape)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(images[0])
    print(labels[0])
    plt.show()
