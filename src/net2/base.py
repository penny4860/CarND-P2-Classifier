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
    
    def _run_single_batch(batch_x, batch_y, pl_x, pl_y, optimize_op, loss_op):
        _, cost_val = sess.run([optimize_op, loss_op],
                               feed_dict={pl_x: batch_x,
                                          pl_y: batch_y})
        return cost_val

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
            total_cost = 0
            for offset, end in get_batch_index(len(X_train), batch_size):
                cost_val = _run_single_batch(X_train[offset:end],
                                             y_train[offset:end],
                                             model.X,
                                             model.Y,
                                             optimizer,
                                             model.loss_op)
                total_cost += cost_val
            _print_cost(epoch, total_cost / total_batch, sess.run(global_step))
            
            evaluate(model, X_train, y_train, sess, batch_size=batch_size)

            if ckpt:
                _save(sess, ckpt, global_step)
        
        print('Training done')

def evaluate(model, images, labels, session=None, ckpt=None, batch_size=100):
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

