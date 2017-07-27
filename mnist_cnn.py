# 이미지 처리 분야에서 가장 유명한 신경망 모델인 CNN 을 이용하여 더 높은 인식률을 만들어봅니다.

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=False)

from src.net2.base import MnistCnn, train, evaluate

train_images = mnist.train.images.reshape(-1, 28, 28, 1)
test_images = mnist.test.images.reshape(-1, 28, 28, 1)

model = MnistCnn()
train(model, train_images, mnist.train.labels)

evaluate(model, test_images, mnist.test.labels, ckpt='models')

