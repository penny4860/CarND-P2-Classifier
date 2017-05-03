

import utils
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models import cnn_model_batch_norm

BATCH_SIZE = 120

if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    # (N, 32, 32, 3), (N, 1)
    images, labels, images_val, labels_val = utils.load_dataset(FILENAME)
    print(labels.shape)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')
 
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])
 
    # Predict
    y_pred = cnn_model_batch_norm(x)
 
    loss = slim.losses.softmax_cross_entropy(y_pred, y)
 
    lr = 0.001
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
 
 
    # Get accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
 
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})
 
    # Training cycle
    total_batch = int(len(images) / BATCH_SIZE)
 
    # Loop for epoch
    for epoch in range(10):
 
        # Random shuffling
 
        # Loop over all batches
        for i in range(total_batch):
 
            # Compute the offset of the current minibatch in the data.
            offset = (i * BATCH_SIZE) % len(images)
            batch_xs = images[offset:(offset + BATCH_SIZE), :, :, :]
            batch_ys = labels[offset:(offset + BATCH_SIZE), :]
 
            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy = sess.run([train_step, accuracy] , feed_dict={x: batch_xs, y: batch_ys, is_training: True})
 
            # Display logs
            print("Epoch:", '%04d,' % (epoch + 1),
            "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))
 
#             # Save the current model if the maximum accuracy is updated
#             if validation_accuracy > max_acc:
#                 max_acc = validation_accuracy
#                 save_path = saver.save(sess, MODEL_DIRECTORY)
#                 print("Model updated and saved in file: %s" % save_path)
 
    print("Optimization Finished!")

    