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


    import net
    cls = net.ConvNetBatchNorm(BATCH_SIZE)
    cls.train(data_train, data_val, "model.ckpt", 5)
 
#             # Save the current model if the maximum accuracy is updated
#             if validation_accuracy > max_acc:
#                 max_acc = validation_accuracy
#                 save_path = saver.save(sess, MODEL_DIRECTORY)
#                 print("Model updated and saved in file: %s" % save_path)
 
    print("Optimization Finished!")

    