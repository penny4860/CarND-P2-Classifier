# -*- coding: utf-8 -*-

import utils
import net
import tensorflow as tf

BATCH_SIZE = 120

if __name__ == '__main__':
    FILENAME = "dataset/train_32x32.mat"
    # (N, 32, 32, 3), (N, 1)

    # 1. with BatchNorm : 90% in 5 epoches
    tf.reset_default_graph()
    cls = net.ConvNetBatchNorm(BATCH_SIZE)
    data_train, data_val = utils.load_dataset(FILENAME)
    cls.train(data_train, data_val, "model2.ckpt", 5)

    # 2. without BatchNorm : 75% in 5 epoches
    tf.reset_default_graph()
    cls = net.ConvNet(BATCH_SIZE)
    data_train, data_val = utils.load_dataset(FILENAME)
    cls.train(data_train, data_val, "model1.ckpt", 5)

#             # Save the current model if the maximum accuracy is updated
#             if validation_accuracy > max_acc:
#                 max_acc = validation_accuracy
#                 save_path = saver.save(sess, MODEL_DIRECTORY)
#                 print("Model updated and saved in file: %s" % save_path)
 
    print("Finished!")

    