# -*- coding: utf-8 -*-

import utils
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models import cnn_model_batch_norm

BATCH_SIZE = 120

if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    # (N, 32, 32, 3), (N, 1)
    #images, labels, images_val, labels_val = utils.load_dataset(FILENAME)
    data_train, data_val = utils.load_dataset(FILENAME)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')
 
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])
 
    # Predict
    y_pred = cnn_model_batch_norm(x, is_training)
 
    cost_op = slim.losses.softmax_cross_entropy(y_pred, y)
 
    lr = 0.001
    train_op = tf.train.AdamOptimizer(lr).minimize(cost_op)
 
 
    # Get accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
 
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
 
    # Loop for epoch
    print ("Train Start!!")
    n_epoch = 5
    # Todo : batch 로 나누는 구조
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

        epoch = 0
        while data_train.epochs_completed < n_epoch:
            batch_xs, batch_ys = data_train.next_batch(BATCH_SIZE)

            _, cost_value = sess.run([train_op, cost_op], feed_dict={x: batch_xs, y: batch_ys, is_training: True})
            # print ("train cost: ", cost_value)

            if epoch != data_train.epochs_completed:
                epoch += 1
                cost_value = sess.run(cost_op, feed_dict={x: data_val.images, y: data_val.labels, is_training: False})
                accuracy = sess.run(accuracy_op, feed_dict={x: data_val.images, y: data_val.labels, is_training: False})
                
                print ("{}-epoch completed. validation cost : {}, {}".format(data_train.epochs_completed, cost_value, accuracy))
        # saver.save(sess, save_file)
 
#             # Save the current model if the maximum accuracy is updated
#             if validation_accuracy > max_acc:
#                 max_acc = validation_accuracy
#                 save_path = saver.save(sess, MODEL_DIRECTORY)
#                 print("Model updated and saved in file: %s" % save_path)
 
    print("Optimization Finished!")

    