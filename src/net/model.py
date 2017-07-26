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
        ys = self.sess.run(self.Y_pred, feed_dict={self.X: X, self._is_training: False})
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
        # 1. load save vars
        from tensorflow.python import pywrap_tensorflow
        reader = pywrap_tensorflow.NewCheckpointReader(load_file)
        saved_variable_names = list(reader.get_variable_to_shape_map().keys())
        saved_variables = []

        # 2. Get the variables by name
        for n in saved_variable_names:
            saved_variables += tf.contrib.framework.get_variables_by_name(n)

        # 3. Init from saved file
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(load_file, saved_variables)
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
            accuracy = self.evaluate(val_set.images, val_set.labels)
            print ("Network parameter loaded. validation accuracy : {}".format(accuracy))
            max_acc = accuracy
        else:
            max_acc = 0

        epoch = 0
        while train_set.epochs_completed < n_epoch:
            batch_xs, batch_ys = train_set.next_batch(self.batch_size)

            _, cost_val = self.sess.run([train_op, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys, self._is_training: True})
            # print ("train cost: ", cost_val)

            if epoch != train_set.epochs_completed:
                epoch += 1
                accuracy = self.evaluate(val_set.images, val_set.labels)
                print ("{}-epoch completed. validation accuracy : {}".format(train_set.epochs_completed, accuracy))
                
                # Save the current model if the maximum accuracy is updated
                if max_acc < accuracy and save_file:
                    max_acc = accuracy
                    saver.save(self.sess, save_file)
                    print("Model updated and saved in file: %s" % save_file)
