import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import *

from config import Config
from tensorflow.examples.tutorials.mnist import input_data


"""
mnist可视化
将minst数据集由二进制文件转化为图片
"""

mnist = input_data.read_data_sets("./mnist/", one_hot=True)

train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

PATH = Config.MNIST_PATH
if not os.path.exists(PATH):
    os.mkdir(PATH)

def mnist_to_images(images,labels):
    classes = np.argmax(labels,1)
    lenght = images.shape[0]

    for i in tqdm(range(lenght)):
        image = images[i,:]*255
        image = image.reshape([28,28])
        mnist_class = classes[i]
        save_path = os.path.join(PATH,str(mnist_class))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path,str(i)+'.jpg')
        cv2.imwrite(save_path,image)

mnist_to_images(train_images,train_labels)


print("finish")
