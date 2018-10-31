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
import skimage

ont_hot = config.ONE_HOT

not_in = []
def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))


def bbbox_to_distance(point_i,point_j):
    if len(point_i) == 4:
        print_word_point = (point_i[2], (point_i[1] + point_i[3]) / 2)
    else:
        print_word_point = ((point_i[2] + point_i[4]) / 2, (point_i[3] + point_i[5]) / 2)

    if len(point_j) == 4:
        hand_word_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
    else:
        hand_word_ponit = ((point_j[0] + point_j[6]) / 2, (point_j[1] + point_j[7]) / 2)

    distences = get_distance(hand_word_ponit, print_word_point)

    return distences


def in_same_line(print_bbox,hand_bbox):
    if len(print_bbox) == 4:
        centre_print_bbox = (print_bbox[1] + print_bbox[3]) / 2
    else:
        centre_print_bbox =  (print_bbox[3] + print_bbox[5]) / 2

    if len(hand_bbox) == 4:
        if (print_bbox[1] >= hand_bbox[1] and print_bbox[3]<= hand_bbox[3]) or (hand_bbox[1]>= print_bbox[1] and hand_bbox[3]<=print_bbox[3]):
            return 'in'

        if centre_print_bbox>hand_bbox[1] and centre_print_bbox<hand_bbox[3]:
            return 'in'
        else:
            return 'out'
    else:
        if centre_print_bbox > hand_bbox[1] and centre_print_bbox < hand_bbox[7]:
            return 'in'
        else:
            return 'out'


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


def draw_pair(pairs,result_list1,result_list2,img,color):
    for pair in pairs:
        reuslt1 = result_list1[pair]
        result2 = result_list2[pairs[pair]]

        cv2.line(img, reuslt1.centre, result2.centre, color, 2)
        cv2.circle(img, reuslt1.centre, radius=20, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, result2.centre, radius=20, color=(0, 255, 0), thickness=-1)


    # cv2.line(img, get_right(bbox_print.bbox), get_left(bbox_hand.bbox), (0, 0, 0), 2)



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
                if ont_hot.get(char) == None:
                    print(char)
                value.append(ont_hot.get(char))

        shape = np.array([batch_size,max_length],dtype=np.int32)
        index = np.array(index,dtype=np.int32)
        try:
            value = np.array(value,dtype=np.int32)
        except:
            # print(label_list)
            pass

        return [index,value,shape]

    def image_normal(self,image):
        if image.shape[0]!=32:
            image = cv2.resize(image,(int(image.shape[1]/image.shape[0]*32),32))
        if image.shape[1] < 10:
            image = cv2.resize(image, (10, 32))
        if image.shape[1]>250:
            image = cv2.resize(image,(250,32))
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


    def data_enhance(self,img):
        img_list = []
        img1 = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)*255   #高斯噪声
        img1 = np.asarray(img1,np.uint8)
        img1 = self.image_normal(img1)
        img2 = skimage.util.random_noise(img, mode='salt', seed=None, clip=True)*255   #椒盐噪声
        img2 = np.asarray(img2, np.uint8)
        img2 = self.image_normal(img2)
        img_list.append(img1)
        img_list.append(img2)
        return img_list


    def get_imges(self,images_path):
        batch_size = len(images_path)
        image_list = []
        max_wide = 0
        images_wide = []

        for path in images_path:
            image = cv2.imread(path)
            image_enhance = image.copy()
            image = self.image_normal(image)
            # if image.shape[1]>250:
            #     print('a')
            images_wide.append(image.shape[1])
            image_list.append(image)
            if config.DATA_ENHANCE:
                img_list = self.data_enhance(image_enhance)
                img_list.extend(img_list)
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
#         image = cv2.resize(image, (int(image.shape[1] / image.shape[0] * 32), 32))：
#         cv2.imwrite(path.replace('test_data','see'),image)




# fuck()
# test()
# #
# print (os.environ['HOME'])
# dataset = DataSet()
# generator = dataset.train_data_generator(32)
# while True:
#     images, labels, wides,length ,epoch= next(generator)
#     if images.shape[2]>250:
#         print(images.shape[2])
#     # print(images.shape[2])
#     if epoch==1:
#         break
    # print('aa')
#
# images, labels, wides,length = dataset.create_val_data()
# print('a')

# dataset = DataSet()
# dataset.analy_data()
# one_hot = []
# for path in tqdm(glob(os.path.join(config.CLEAN_DATA,'*'))):
#     label = path.split('_')[-1]
#     for char in label:
#         one_hot.append(char)
#
# one_hot = list(set(one_hot))
# print(one_hot)
