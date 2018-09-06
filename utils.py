import os
from glob import glob
import os
import cv2
import sys
import math
import matplotlib.pyplot as plot
import matplotlib
import numpy as np
import os
import random
from config import Config as config

ont_hot = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'+':10,'-':11,'=':12,'×':13,'÷':14,'(':15,')':16}


class DataSet(object):
    def __init__(self,noise_able = True):
        clean_data = glob(os.path.join(config.CLEAN_DATA,'*'))
        noise_data = glob(os.path.join(config.NOISE_DATA,'*'))
        self.val_data = glob(os.path.join(config.VAL_DATA,'*'))
        self.all_data = []
        if noise_able:
            self.all_data = noise_data
        else:
            self.all_data = clean_data

    def list_to_sparse(self,label_list):
        index = []
        value = []
        max_length = 0
        batch_size = len(label_list)

        for x,labels in enumerate(label_list):
            if len(labels)>max_length:
                max_length = len(labels)
            for y,char in enumerate(labels):
                index.append([x,y])
                value.append(ont_hot.get(char))

        shape = np.array([batch_size,max_length],dtype=np.int32)
        index = np.array(index,dtype=np.int32)
        value = np.array(value,dtype=np.int32)

        return [index,value,shape]

    def image_normal(self,image):
        if image.shape[0]!=32:
            image = cv2.resize(image,(int(image.shape[1]/image.shape[0]*32),32))
        if image.shape[1]>250:
            image = cv2.resize(image, (250, 32))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = image/255*2-1
        return image

    def get_labels(self,images_path):
        label_list = []
        label_len = []

        target_input = np.ones((len(images_path), config.SEQ_MAXSIZE), dtype=np.float32) + 2
        target_out = np.ones((len(images_path), config.SEQ_MAXSIZE), dtype=np.float32) + 2

        for path in images_path:
            label = path.split('_')[-1].replace('.jpg', '')
            label = label.replace('.png', '')
            label_list.append(label)
            label_len.append(len(label))

        for i in range(len(label_list)):
            # seq_len.append(len(label[i]))
            target_input[i][0] = 0  # 第一个为GO
            for j in range(len(label_list[i])):
                target_input[i][j + 1] = config.ONE_HOT[label_list[i][j]]
                target_out[i][j] = config.ONE_HOT[label_list[i][j]]
            target_out[i][len(label_list[i])] = 1
        return target_input, target_out,label_list


    def get_imges(self,images_path):
        batch_size = len(images_path)
        image_list = []
        max_wide = config.IMG_MAXSIZE
        images_wide = []

        for path in images_path:
            image = cv2.imread(path)
            image = self.image_normal(image)
            images_wide.append(image.shape[1])
            image_list.append(image)


        images = np.zeros([batch_size,config.IMAGE_HEIGHT,max_wide])

        for i,image in enumerate(image_list):
            images[i,:,0:image.shape[1]] = image
        images = images[...,np.newaxis]

        wides = np.array(images_wide,dtype=np.int32)

        return images,wides


    def train_data_generator(self,batch_size):
        all_data = self.all_data
        step = 0
        epoch = 0
        while True:
            if (step+1)*batch_size >len(all_data):
                random.shuffle(all_data)
                step=0
                epoch = epoch+1
            images_path = all_data[step*batch_size:(step+1)*batch_size]
            images, _ = self.get_imges(images_path)
            target_input, target_out,label_list = self.get_labels(images_path)

            step = step+1

            yield images,target_input, target_out,label_list,epoch

    def create_val_data(self):
        val_data = self.val_data
        images,  _ = self.get_imges(val_data)
        target_input, target_out, label_list = self.get_labels(val_data)
        return images,target_input, target_out,label_list


# def fuck():
#     val_data = glob(os.path.join(config.VAL_DATA, '*'))
#     for path in val_data:
#         img = cv2.imread(path)
#         cut = 0
#         for i in range(300):
#             if img[i,0,0]==255:
#                 cut = i
#                 break
#         img = img[0:cut,:,:]
#         cv2.imwrite(path.replace('test_data','test_data2'),img)
#         print("a")
#
# def test():
#     val_data = glob(os.path.join(config.VAL_DATA, '*'))
#     for path in val_data:
#         image = cv2.imread(path)
#         image = cv2.resize(image, (int(image.shape[1] / image.shape[0] * 32), 32))
#         cv2.imwrite(path.replace('test_data','see'),image)




# fuck()
# test()
# #
# print (os.environ['HOME'])
# dataset = DataSet()
# generator = dataset.train_data_generator(config.BATCH_SIZE)
# while True:
#     images, target_input, target_out,epoch = next(generator)
#     print('aa')
#
# images, labels, wides,length = dataset.create_val_data()
# print('a')