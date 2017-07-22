# -*- coding: utf-8 -*-

"""Sample Models to train small image patches"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
from .model import _Model

class ConvNetBatchNorm(_Model):
    def build(self):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.float32, [None, 10])

        batch_norm_params = {'is_training': self._is_training,
                             'decay': 0.9,
                             'updates_collections': None}

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

            net = slim.conv2d(X, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            net = slim.fully_connected(net, 1024, scope='fc3')
            Y_pred = slim.fully_connected(net, 10, activation_fn=None,
                                          normalizer_fn=None, scope='fco')

        return X, Y, Y_pred

    def cost(self):
        # cost = tf.reduce_mean(tf.square(tf.subtract(self.Y_pred, self.Y)))
        cost_op = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_pred,
                                                          labels=self.Y)
        return cost_op


class ConvNet(_Model):
    def build(self):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.float32, [None, 10])

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=None):

            net = slim.conv2d(X, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            net = slim.fully_connected(net, 1024, scope='fc3')
            Y_pred = slim.fully_connected(net, 10, activation_fn=None,
                                          normalizer_fn=None, scope='fco')

        return X, Y, Y_pred

    def cost(self):
        # cost = tf.reduce_mean(tf.square(tf.subtract(self.Y_pred, self.Y)))
        cost_op = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_pred,
                                                          labels=self.Y)
        return cost_op