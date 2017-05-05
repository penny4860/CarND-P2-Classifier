# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim


class _Model:
    """
    # Attributes
        X : tf.placeholder
        Y : tf.placeholder
        Y_pred : tf.Tensor
        batch_size : int
    """
    def __init__(self, batch_size=1):
        self._is_training = tf.placeholder(tf.bool, name='MODE')
        self.X, self.Y, self.Y_pred = self.build()
        self.batch_size = batch_size
        self.sess = tf.Session()

    def build(self):
        raise NotImplementedError

    def inference(self, X):
        """
        # Args
            sess : tf.Session()
            X : np.array
        """
        ys = self.sess.run(self.Y_pred, feed_dict={self.X: X})
        return ys

    def cost(self):
        raise NotImplementedError

    def evaluate(self, images, labels):
        """
        # Args
            images : (N, n_rows, n_cols, n_ch)
            labels : (N, n_categories)
        # Returns
            accuracy : float
        """
        # Get accuracy of model
        correct_prediction = tf.equal(tf.argmax(self.Y_pred, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = self.sess.run(accuracy_op, feed_dict={self.X: images, self.Y: labels, self._is_training: False})
        return accuracy

    def load_params(self, load_file):
        """network parameter 를 load 하는 함수. """
        
        # Variable 중에서 naming 이 weights, biases 로 되어있는 Variable 객체만 file 에 저장되어있는 value 로 restore.
        weights = tf.contrib.framework.get_variables_by_name('weights')
        biases = tf.contrib.framework.get_variables_by_name('biases')
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(load_file, weights+biases)
        self.sess.run(init_assign_op, init_feed_dict)


    def train(self, train_set, val_set, n_epoch, save_file=None, load_file=None):
        saver = tf.train.Saver()
        cost = self.cost()
        optimizer = tf.train.AdamOptimizer(0.001, name="ADAM")
        train_op = optimizer.minimize(cost)
        
        print ("Train Start!!")
        # Todo : batch 로 나누는 구조
        self.sess.run(tf.global_variables_initializer())
        if load_file:
            self.load_params(load_file)

        epoch = 0
        while train_set.epochs_completed < n_epoch:
            batch_xs, batch_ys = train_set.next_batch(self.batch_size)

            _, cost_val = self.sess.run([train_op, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys, self._is_training: True})
            # print ("train cost: ", cost_val)

            if epoch != train_set.epochs_completed:
                epoch += 1
                accuracy = self.evaluate(val_set.images, val_set.labels)
                print ("{}-epoch completed. validation accuracy : {}".format(train_set.epochs_completed, accuracy))
        if save_file:
            saver.save(self.sess, save_file)
