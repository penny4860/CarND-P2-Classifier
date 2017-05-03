# -*- coding: utf-8 -*-

import src.utils as utils
import src.net as net
import tensorflow as tf

BATCH_SIZE = 120

if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    # (N, 32, 32, 3), (N, 1)

    # 1. Train
    tf.reset_default_graph()
    cls = net.ConvNet(BATCH_SIZE)
    data_train, data_val = utils.load_dataset(FILENAME)
    cls.train(data_train, data_val, 5, "model/model1.ckpt")

    # 2. Train model using initialized from saved parameters
    tf.reset_default_graph()
    cls = net.ConvNet(BATCH_SIZE)
    data_train, data_val = utils.load_dataset(FILENAME)
    cls.train(data_train, data_val, 5, "model/model2.ckpt", "model/model1.ckpt")

    print("Finished!")

    