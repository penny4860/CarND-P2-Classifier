# -*- coding: utf-8 -*-

import src.utils as utils
from src.net.tiny_model import ConvNetBatchNorm, ConvNet
import tensorflow as tf

BATCH_SIZE = 120

if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    # (N, 32, 32, 3), (N, 1)

    # 1. with BatchNorm : 90% in 5 epoches
    tf.reset_default_graph()
    cls = ConvNetBatchNorm(BATCH_SIZE)
    data_train, data_val = utils.load_dataset(FILENAME)
    cls.train(data_train, data_val, 5)

    # 2. without BatchNorm : 75% in 5 epoches
    tf.reset_default_graph()
    cls = ConvNet(BATCH_SIZE)
    data_train, data_val = utils.load_dataset(FILENAME)
    cls.train(data_train, data_val, 5)

    print("Finished!")

    