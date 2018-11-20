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
from PIL import Image,ImageDraw,ImageFont

ont_hot = config.ONE_HOT

not_in = []

def area(point1,point2):
    return max((point2[0]-point1[0]),0)*max((point2[1]-point1[1]),0)

def get_iou(box1,box2):
    area1 = area((box1[0],box1[1]),(box1[2],box1[3]))
    area2 = area((box2[0],box2[1]),(box2[2],box2[3]))

    point1 = (max(box1[0],box2[0]),max(box1[1],box2[1]))
    point2 = (min(box1[2],box2[2]),min(box1[3],box2[3]))

    area3 = area(point1,point2)

    return area3/(area1+area2-area3)

def image_size_normal(img):
    '''
    将图片统一resize成4k,并返回其缩放比例

    :param img:     resize的图片
    :return:    图片的缩放比例
    '''
    x_pro = 3024 / img.shape[1]
    y_pro = 4031 / img.shape[0]
    img = cv2.resize(img, (3024, 4032))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img,x_pro,y_pro

def eval_label(label):
    '''
    判断字符串中的算式属于哪种状态
    1.正确
    2.错误
    3.有问题

    :param label:   算式的字符串
    :return:    算式的状态
    '''


    try:
        if '=' not in label or label=='':
            return 'problem'
        else:
            left = label.split('=')[0]
            right = label.split('=')[1]

        if right=='' or left == '':
            return 'problem'

        left = left.replace('×', '*')
        if '÷' in left and ('*' in right or '~' in right):
            left1 = left.replace('÷', '//')
            left2 = left.replace('÷', '%')
            left1 = eval(left1)
            left2 = eval(left2)

            if '*' in right or '~' in right:
                right1 = ''
                right2 = ''
                if '*' in right:
                    right1 = right.split('*')[0]
                    right2 = right.split('*')[-1]


                if '~' in right:
                    right1 = right.split('~')[0]
                    right2 = right.split('~')[-1]

                right1 = eval(right1)
                right2 = eval(right2)

                if right1==int(left1) and right2 == int(left2):
                    return 'right'

                else:
                    return 'error'

            else:
                if left2 == 0:
                    if left1 == int(right):
                        return 'right'
                    else:
                        return 'error'
                else:
                    return 'problem'
        else:
            if '÷' in left:
                left = left.replace('÷', '/')
            result = eval(left)
            if result == int(right):
                return 'right'
            else:
                return 'error'
    except:
        return 'problem'


def draw_bboxes(img,all_result,x_pro,y_pro,display_all = True):
    for result in all_result:
        if result.type == 'fraction':
            rgb = (0, 255, 255)
            cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                          (int(result.right * x_pro), int(result.bottom * y_pro)), rgb, 4)
            continue
        if result.state == 'right':
            rgb = (0,255,0)
            cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                          (int(result.right * x_pro), int(result.bottom * y_pro)), rgb, 4)

        elif result.state == 'error':
            rgb = (255, 0, 0)
            cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                          (int(result.right * x_pro), int(result.bottom * y_pro)), rgb, 4)

        elif result.state == 'problem' :
            if display_all:
                rgb = (0, 0, 255)
                cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                              (int(result.right * x_pro), int(result.bottom * y_pro)), rgb, 4)



        else:
            rgb = (0, 0, 0)
            if result.type == 'print':
                rgb = (0, 255, 0)
            elif result.type == 'hand':
                rgb = (255, 0, 0)
            elif result.type == 'merge':
                rgb = (0, 0, 255)
            cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                          (int(result.right * x_pro), int(result.bottom * y_pro)), rgb, 4)




def draw_result(img,all_result,x_pro,y_pro):
    ttfont = ImageFont.truetype('SimSun.ttf',50)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for result in all_result:
        if result.state != 'problem':
            draw.text((int(result.left*x_pro),int(result.top*y_pro-50)),result.output,fill='blue',font=ttfont)
        else:
            draw.text((int(result.left*x_pro),int(result.top*y_pro-50)),result.output,fill='blue',font=ttfont)
    return img


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


def draw_pair(pairs,result_list1,result_list2,img,color,x_pro=1, y_pro=1):
    for pair in pairs:
        reuslt1 = result_list1[pair]
        result2 = result_list2[pairs[pair]]

        centre1 = (int(reuslt1.centre[0]*x_pro),int(reuslt1.centre[1]*y_pro))
        centre2 = (int(result2.centre[0] * x_pro), int(result2.centre[1] * y_pro))

        cv2.line(img, centre1, centre2, color, 3)
        cv2.circle(img, centre1, radius=15, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, centre2, radius=15, color=(0, 255, 0), thickness=-1)


def draw_column_pair(column_pairs,cell_list,img,x_pro=1, y_pro=1):
    for top in column_pairs:
        top_cell = cell_list[top]
        if column_pairs[top] == -1:
            continue
        bottom_cell = cell_list[column_pairs[top]]

        top_cell_centre = (int(top_cell.centre[0]*x_pro),int(top_cell.centre[1]*y_pro))
        bottom_cell_centre = (int(bottom_cell.centre[0]*x_pro),int(bottom_cell.centre[1]*y_pro))

        cv2.line(img, top_cell_centre, bottom_cell_centre, (255, 0, 0),3)
        cv2.circle(img, top_cell_centre, radius=15, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, bottom_cell_centre, radius=15, color=(0, 255, 0), thickness=-1)





class DataSet(object):
    def __init__(self,train_data_list = config.TRAIN_DATA_LIST,val_data_list =config.VAL_DATA_LIST):
        self.train_data_list = [glob(os.path.join(train_data,'*')) for train_data in train_data_list]
        self.val_data_list = [glob(os.path.join(val_data,'*')) for val_data in val_data_list]

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
            label = label.replace('.JPG', '')
            label_list.append(label)
            label_len.append(len(label))

        labels = self.list_to_sparse(label_list.copy()),
        label_len = np.array(label_len, dtype=np.int32)
        return  labels[0], label_len,label_list


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
        all_data_list= self.train_data_list
        step = [0]*len(all_data_list)
        epoch = [0]*len(all_data_list)
        while True:
            images_path = []
            for i,all_data in enumerate(all_data_list):
                if (step[i]+1)*batch_size >len(all_data):
                    random.shuffle(all_data)
                    step[i]=0
                    epoch[i] = epoch[i]+1
                images_path.extend(all_data[step[i]*batch_size:(step[i]+1)*batch_size])
                step[i] = step[i]+1
            images, wides = self.get_imges(images_path)
            labels, length,real_labels = self.get_labels(images_path)
            # if wides[0]<10:
            #     print(images_path)



            yield images, labels, wides,length,real_labels,epoch

    def create_val_data(self):
        val_data_list = self.val_data_list
        val_data = []
        for data in val_data_list:
            val_data.extend(data)
        all_val_data = []
        i = 0
        while i*config.BATCH_SIZE<len(val_data):
            if (i+1)*config.BATCH_SIZE>len(val_data):
                end = len(val_data)
            else:
                end = (i+1)*config.BATCH_SIZE
            images,  wides = self.get_imges(val_data[i*config.BATCH_SIZE:end])
            labels,length ,real_labels= self.get_labels(val_data[i*config.BATCH_SIZE:end])
            all_val_data.append((images, labels, wides, length,real_labels))
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



if __name__ == '__main__':

    # fuck()
    # test()
    # #
    # print (os.environ['HOME'])
    dataset = DataSet()
    generator = dataset.train_data_generator(32)
    dataset.create_val_data()
    while True:
        images, labels, wides,length ,epoch= next(generator)
        if images.shape[2]>250:
            print(images.shape[2])
        print(epoch)

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
