# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model:
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
        return None

    def inference(self, X):
        """
        # Args
            sess : tf.Session()
            X : np.array
        """
        ys = self.sess.run(self.Y_pred, feed_dict={self.X: X})
        return ys

    def cost(self):
        pass

    def load_params(self, save_file):
        """Model.sess 에 save_file 에 저장되어있는 parameter 를 restore 하는 함수."""

        # The file path to save the data
        saver = tf.train.Saver()
        saver.restore(self.sess, save_file)

    def train(self, train_set, val_set, save_file, n_epoch, load_file=None):
        saver = tf.train.Saver()
        cost = self.cost()
        optimizer = tf.train.AdamOptimizer(0.001, name="ADAM")
        train_op = optimizer.minimize(cost)
        
        # Get accuracy of model
        correct_prediction = tf.equal(tf.argmax(self.Y_pred, 1), tf.argmax(self.Y, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        print ("Train Start!!")
        # Todo : batch 로 나누는 구조
        if load_file:
            self.sess.run(tf.global_variables_initializer())
            # variables_to_restore = slim.get_variables_to_restore(include=["conv1"], exclude=["ADAM:0", "ADAM_1:0"])
            #variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
            
            weights = tf.contrib.framework.get_variables_by_name('weights')
            biases = tf.contrib.framework.get_variables_by_name('biases')
            
            for var in weights+biases:
                print(var.name)
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(load_file, weights+biases)
            self.sess.run(init_assign_op, init_feed_dict)
        else:
            self.sess.run(tf.global_variables_initializer())

        epoch = 0
        while train_set.epochs_completed < n_epoch:
            batch_xs, batch_ys = train_set.next_batch(self.batch_size)

            _, cost_val = self.sess.run([train_op, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys, self._is_training: True})
            # print ("train cost: ", cost_val)

            if epoch != train_set.epochs_completed:
                epoch += 1
                accuracy = self.sess.run(accuracy_op, feed_dict={self.X: val_set.images, self.Y: val_set.labels, self._is_training: False})
                print ("{}-epoch completed. validation accuracy : {}".format(train_set.epochs_completed, accuracy))
        saver.save(self.sess, save_file)


class ConvNetBatchNorm(Model):
    def build(self):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.float32, [None, 10])

        batch_norm_params = {'is_training': self._is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
    
            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(X, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')
    
            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            Y_pred = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')

        return X, Y, Y_pred

    def cost(self):
        # cost = tf.reduce_mean(tf.square(tf.subtract(self.Y_pred, self.Y)))
        cost_op = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_pred, labels=self.Y)
        return cost_op

class ConvNet(Model):
    def build(self):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        Y = tf.placeholder(tf.float32, [None, 10])

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=None):
    
            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(X, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')
    
            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            Y_pred = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')

        return X, Y, Y_pred

    def cost(self):
        # cost = tf.reduce_mean(tf.square(tf.subtract(self.Y_pred, self.Y)))
        cost_op = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_pred, labels=self.Y)
        return cost_op

