# Load pickled data
import pickle
import numpy as np
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from sklearn.utils import shuffle


def load_dataset(files = ["../dataset/train.p",
                          "../dataset/valid.p",
                          "../dataset/test.p"]):
    training_file = files[0]
    validation_file= files[1]
    testing_file = files[2]

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def preprocess(images):
    return (images.astype(float) - 128)/128

########################################################################### base
import tensorflow as tf
from abc import ABCMeta, abstractmethod
from sklearn.utils import shuffle

class _Model(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self.X = self._create_input_placeholder()
        self.Y = self._create_output_placeholder()
        self.is_training = self._create_is_train_placeholder()

        self.inference_op = self._create_inference_op()
        self.loss_op = self._create_loss_op()
        self.accuracy_op = self._create_accuracy_op()

    @abstractmethod
    def _create_input_placeholder(self):
        return tf.placeholder(tf.float32, [None, 28, 28, 1])

    @abstractmethod
    def _create_inference_op(self):
        pass

    @abstractmethod
    def _create_loss_op(self):
        one_hot_y = tf.one_hot(self.Y, 10)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=one_hot_y))

    def _create_output_placeholder(self):
        return tf.placeholder(tf.int64, [None])

    def _create_accuracy_op(self):
        is_correct = tf.equal(tf.argmax(self.inference_op, 1), self.Y)
        return tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def _create_is_train_placeholder(self):
        is_training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                  shape=(),
                                                  name='is_training')
        return is_training


def train(model, X_train, y_train, X_val, y_val, batch_size=100, n_epoches=5, ckpt=None):

    def _run_single_epoch(X_train, y_train, batch_size):
        total_cost = 0
        for offset, end in get_batch_index(len(X_train), batch_size):
            _, cost_val = sess.run([optimizer, model.loss_op],
                                   feed_dict={model.X: X_train[offset:end],
                                              model.Y: y_train[offset:end],
                                              model.is_training: True})
            total_cost += cost_val
        return total_cost
   
    def _save(sess, ckpt, global_step):
        import os
        directory = os.path.dirname(ckpt)
        if not os.path.exists(directory):
            os.mkdir(directory)
            
        saver = tf.train.Saver()
        saver.save(sess, ckpt, global_step=global_step)
        # saver.save(sess, 'models/cnn')
        # saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)

    def _print_cost(epoch, cost, global_step):
        print('Epoch: {:3d}, Training Step: {:5d}, Avg. cost ={:.3f}'.format(epoch + 1, global_step, cost))
    
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(0.001).minimize(model.loss_op, global_step=global_step)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = get_n_batches(len(X_train), batch_size)
        
        for epoch in range(n_epoches):
            X_train, y_train = shuffle(X_train, y_train)
            cost = _run_single_epoch(X_train, y_train, batch_size)
            _print_cost(epoch, cost / total_batch, sess.run(global_step))
            
            evaluate(model, X_train, y_train, sess, batch_size=batch_size)
            evaluate(model, X_val, y_val, sess, batch_size=batch_size)

            if ckpt:
                _save(sess, ckpt, global_step)
        
        print('Training done')

def evaluate(model, images, labels, session=None, ckpt=None, batch_size=100):
    """
    ckpt : str
        ckpt directory or ckpt file
    """
    def _evaluate(sess):
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt))

        accuracy_value = 0
        for offset, end in get_batch_index(len(images), batch_size):
            accuracy_value += sess.run(model.accuracy_op,
                                      feed_dict={model.X: images[offset:end],
                                                 model.Y: labels[offset:end],
                                                 model.is_training: False})
        accuracy_value = accuracy_value / get_n_batches(len(images), batch_size)
        return accuracy_value

    if session:
        accuracy = _evaluate(session)
    else:
        sess = tf.Session()
        accuracy = _evaluate(sess)
        sess.close()
        
    print('Accuracy: {:.4f}'.format(accuracy))
    return accuracy


def get_batch_index(num_examples, batch_size=100):
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        if end > num_examples:
            break
        yield (offset, end)

def get_n_batches(num_examples, batch_size=100):
    return int(num_examples / batch_size)

################################################################################################################

class SignModel(_Model):
#     Accuracy:  0.975774
#     Accuracy:  0.800907
    def _create_input_placeholder(self):
        return tf.placeholder(tf.float32, [None, 32, 32, 3])

    def _create_inference_op(self):
        W1_1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
        L1 = tf.nn.conv2d(self.X, W1_1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        W1_2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
        L1 = tf.nn.conv2d(L1, W1_2, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2_1 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2_1, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        W2_2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L2, W2_2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W3 = tf.Variable(tf.random_normal([8 * 8 * 64, 256], stddev=0.01))
        L3 = tf.reshape(L2, [-1, 8 * 8 * 64])
        L3 = tf.matmul(L3, W3)
        L3 = tf.nn.relu(L3)
        
        W4 = tf.Variable(tf.random_normal([256, 43], stddev=0.01))
        model = tf.matmul(L3, W4)
        return model

    def _create_loss_op(self):
        one_hot_y = tf.one_hot(self.Y, 43)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=one_hot_y))

# 1. load dataset
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()

# 2. preprocess images
X_train_prep, X_test_prep, X_valid_prep = preprocess(X_train), preprocess(X_test), preprocess(X_valid)

model = SignModel()
train(model, X_train_prep, y_train, X_valid_prep, y_valid, batch_size=32, n_epoches=20, ckpt='ckpts/cnn')
evaluate(model, X_test_prep, y_test, ckpt='ckpts')




