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
from tqdm import tqdm
import matplotlib.pyplot as plot

ont_hot = config.ONE_HOT

def create_input(image_list,max_wide,wide_list):
    images = np.zeros([len(image_list), config.IMAGE_HEIGHT, max_wide])

    for i, image in enumerate(image_list):
        images[i, :, 0:image.shape[1]] = image
    images = images[..., np.newaxis]

    wides = np.array(wide_list, dtype=np.int32)

    return images,wides

def image_normal(image):
    if image.shape[0] != 32:
        image = cv2.resize(image, (int(image.shape[1] / image.shape[0] * 32), 32))
    if image.shape[1] < 10:
        image = cv2.resize(image,(10, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255 * 2 - 1
    return image

def pre_to_output(sentence):
    sentence = sentence.tolist()
    decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))
    result = ''.join(list(map(lambda x: decode.get(x), sentence[0])))

    return result



class DataSet(object):
    def __init__(self,noise_able = False):
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
        try:
            value = np.array(value,dtype=np.int32)
        except:
            print(label_list)

        return [index,value,shape]

    def image_normal(self,image):
        if image.shape[0]!=32:
            image = cv2.resize(image,(int(image.shape[1]/image.shape[0]*32),32))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = image/255*2-1
        return image

    def get_labels(self,images_path):
        label_list = []
        label_len = []

        for path in images_path:
            label = path.split('_')[-1].replace('.jpg', '')
            label = label.replace('.png', '')
            label_list.append(label)
            label_len.append(len(label))

        labels = self.list_to_sparse(label_list),
        label_len = np.array(label_len, dtype=np.int32)
        return  labels[0], label_len

    def get_imges(self,images_path):
        batch_size = len(images_path)
        image_list = []
        max_wide = 0
        images_wide = []

        for path in images_path:
            image = cv2.imread(path)
            image = self.image_normal(image)
            images_wide.append(image.shape[1])
            image_list.append(image)
            if image.shape[1]>max_wide:
                max_wide = image.shape[1]

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
            images, wides = self.get_imges(images_path)
            labels, length = self.get_labels(images_path)
            # if wides[0]<10:
            #     print(images_path)

            step = step+1

            yield images, labels, wides,length,epoch

    def create_val_data(self):
        val_data = self.val_data
        all_val_data = []
        i = 0
        while i*config.BATCH_SIZE<len(val_data):
            if (i+1)*config.BATCH_SIZE>len(val_data):
                end = len(val_data)
            else:
                end = (i+1)*config.BATCH_SIZE
            images,  wides = self.get_imges(val_data[i*config.BATCH_SIZE:end])
            labels,length = self.get_labels(val_data[i*config.BATCH_SIZE:end])
            all_val_data.append((images, labels, wides, length))
            i = i+1
        return all_val_data

    def analy_data(self):
        all_num = []
        avg_num = 0
        all_len_wid = []
        avg_len_wid = 0
        all_font_len = []
        avg_font_len = 0
        for data in tqdm(self.all_data):
            img = cv2.imread(data)
            label = data.split('_')[-1].replace('.jpg', '')
            label = label.replace('.png', '')
            label_num = len(label)
            avg_num = label_num+avg_num
            all_num.append(label_num)
            img = cv2.resize(img,(int(img.shape[1]/img.shape[0]*32),32))
            len_wid = img.shape[1]/img.shape[0]
            avg_len_wid = avg_len_wid+len_wid
            all_len_wid.append(len_wid)
            font_len = img.shape[1]/label_num
            avg_font_len = avg_font_len + font_len
            all_font_len.append(font_len)

        avg_num = avg_num/len(all_num)
        avg_len_wid = avg_len_wid/len(all_num)
        avg_font_len = avg_font_len/len(all_num)

        plot.figure(figsize=(40, 10), dpi=80)  # 绘制直方图
        plot.xticks(fontsize=40)
        plot.yticks(fontsize=40)
        plot.hist(all_num, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        plot.savefig('all_num_analy', bbox_inches='tight')

        plot.figure(figsize=(40, 10), dpi=80)
        plot.xticks(fontsize=40)
        plot.yticks(fontsize=40)
        plot.hist(all_len_wid, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        plot.savefig('all_len_wid_analy', bbox_inches='tight')

        plot.figure(figsize=(40, 10), dpi=80)  # 绘制直方图
        plot.xticks(fontsize=40)
        plot.yticks(fontsize=40)
        plot.hist(all_font_len, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        plot.savefig('all_font_len_analy', bbox_inches='tight')




        print(avg_num)
        print(avg_len_wid)
        print(avg_font_len)

        plot.show()







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
# generator = dataset.train_data_generator(1)
# while True:
#     images, labels, wides,length ,epoch= next(generator)
#     if epoch==1:
#         break
    # print('aa')
#
# images, labels, wides,length = dataset.create_val_data()
# print('a')

# dataset = DataSet()
# dataset.analy_data()