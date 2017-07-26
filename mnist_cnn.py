# 이미지 처리 분야에서 가장 유명한 신경망 모델인 CNN 을 이용하여 더 높은 인식률을 만들어봅니다.
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


class Model(object):

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.Y = tf.placeholder(tf.float32, [None, 10])

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
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))


def train(model):
    optimizer = tf.train.AdamOptimizer(0.001).minimize(model.loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)
    
    for epoch in range(5):
        total_cost = 0
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)
    
            _, cost_val = sess.run([optimizer, model.loss],
                                   feed_dict={model.X: batch_xs,
                                              model.Y: batch_ys})
            total_cost += cost_val
    
        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
    print('최적화 완료!')
    saver = tf.train.Saver()
    saver.save(sess, 'models/cnn')
    # saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)
    sess.close()

def evaluate(model):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('models'))
    
        is_correct = tf.equal(tf.argmax(model.logits, 1), tf.argmax(model.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도:', sess.run(accuracy,
                                feed_dict={model.X: mnist.test.images.reshape(-1, 28, 28, 1),
                                           model.Y: mnist.test.labels}))


model = Model()
train(model)
evaluate(model)

