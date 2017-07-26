# 이미지 처리 분야에서 가장 유명한 신경망 모델인 CNN 을 이용하여 더 높은 인식률을 만들어봅니다.
import tensorflow as tf
from sklearn.utils import shuffle

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=False)


class Model(object):

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.Y = tf.placeholder(tf.int64, [None])

        self.logits = self._build()
        self.loss = self._loss_operation()

    def _build(self):
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
        L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
        L3 = tf.matmul(L3, W3)
        L3 = tf.nn.relu(L3)
        
        W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
        model = tf.matmul(L3, W4)
        return model

    def _loss_operation(self):
        one_hot_y = tf.one_hot(self.Y, 10)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_y))


def train(model, X_train, y_train):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(model.loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)
    
    num_examples = len(X_train)
    
    for epoch in range(5):
        total_cost = 0
        
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]

            _, cost_val = sess.run([optimizer, model.loss],
                                   feed_dict={model.X: batch_x,
                                              model.Y: batch_y})
            total_cost += cost_val
    
        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
    print('Training done')
    saver = tf.train.Saver()
    saver.save(sess, 'models/cnn')
    # saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)
    sess.close()

def evaluate(model):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    
        is_correct = tf.equal(tf.argmax(model.logits, 1), model.Y)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('Accuracy: ', sess.run(accuracy,
                                feed_dict={model.X: mnist.test.images.reshape(-1, 28, 28, 1),
                                           model.Y: mnist.test.labels}))


train_images = mnist.train.images.reshape(-1, 28, 28, 1)

model = Model()
train(model, train_images, mnist.train.labels)
evaluate(model)

