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


def train(model, X_train, y_train, X_val, y_val, batch_size=100, n_epoches=5, ckpt=None):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(model.loss_op)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = len(X_train) / batch_size
        num_examples = len(X_train)
        
        for epoch in range(n_epoches):
            total_cost = 0
            
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
    
                _, cost_val = sess.run([optimizer, model.loss_op],
                                       feed_dict={model.X: batch_x,
                                                  model.Y: batch_y})
                total_cost += cost_val
        
            print('Epoch:', '%04d' % (epoch + 1),
                  'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
            
            evaluate(model, X_val, y_val, sess)
        
        print('Training done')
        
        if ckpt:
            import os
            directory = os.path.dirname(ckpt)
            if not os.path.exists(directory):
                os.mkdir(directory)
                
            saver = tf.train.Saver()
            saver.save(sess, ckpt)
            # saver.save(sess, 'models/cnn')
            # saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)

def evaluate(model, images, labels, session=None, ckpt=None):
    """
    ckpt : str
        ckpt directory or ckpt file
    """
    # Todo : accuracy op를 batch 별로 실행할 수 있도록 수정
    # sample 숫자가 많으면 memory 문제로 평가가 불가능하다.
    def _evaluate(sess):
        if ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt))
            
        print('Accuracy: ', sess.run(model.accuracy_op,
                                        feed_dict={model.X: images,
                                                   model.Y: labels}))

    if session:
        _evaluate(session)
    else:
        sess = tf.Session()
        _evaluate(sess)
        sess.close()
################################################################################################################

class SignModel(_Model):
    def _create_input_placeholder(self):
        return tf.placeholder(tf.float32, [None, 32, 32, 3])

    def _create_inference_op(self):
        W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
        L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
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
train(model, X_train_prep, y_train, X_valid_prep, y_valid, ckpt='ckpts/cnn')
evaluate(model, X_test_prep, y_test, ckpt='ckpts')



