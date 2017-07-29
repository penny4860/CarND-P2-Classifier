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


def train(model, X_train, y_train, batch_size=100, n_epoches=5, ckpt=None):
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
            
            evaluate(model, X_train, y_train, sess)
        
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

def evaluate(model, images, labels, batch_size=100, session=None, ckpt=None):
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

        accuracy_value = 0
        for offset, end in get_batch_index(len(images), batch_size):
            accuracy_value += sess.run(model.accuracy_op,
                                      feed_dict={model.X: images[offset:end],
                                                 model.Y: labels[offset:end]})
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

