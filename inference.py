from cutdata import ParseXml

from cutdata import ParseXml
import os
from glob import glob
import cv2
from sklearn.metrics.pairwise import pairwise_distances
import math
import from_xml_read_label
import random
from tqdm import tqdm
import numpy as np
from config import Config as config

class Bbox(object):
    def __init__(self,bbox,label,classes):
        self.bbox = bbox
        self.label = label
        self.classes = classes

        if len(bbox) == 4:
            self.top = bbox[1]
            self.bottom = bbox[3]
            self.left = bbox[0]
            self.right = bbox[2]
        else:
            self.top = min(bbox[1],bbox[3])
            self.bottom = max(bbox[5],bbox[7])
            self.left = min(bbox[0],bbox[6])
            self.right = min(bbox[2],bbox[4])

class Img_ALL_BBox(object):
    def __init__(self,img):
        self.print_word = []
        self.hand_word = []
        self.img_path = ''
        self.pair = {}
        self.img = img

    def cut_biger_img(self,path,num=0):
        for i,print_num in enumerate(self.pair):
            hand_num = self.pair[print_num]
            print = self.print_word[print_num]
            hand = self.hand_word[hand_num]
            top = min(print.top,hand.top)
            bottom = max(print.bottom,hand.bottom)
            left = min(print.left,hand.left)
            right = max(print.right,hand.right)

            biger_img = self.img[top:bottom+1,left:right+1,...]

            label = print.label+hand.label
            cv2.imwrite(path+str(num)+'_'+str(i)+'_'+label+'.jpg',biger_img)

def find_labels(name,j):
    name = name.split('.')[0]
    name = name.split('_')[-1]
    path = config.ORIGINAL_TRAIN_DATA
    paths = glob(os.path.join(path,'*'))
    label_path = ''
    for i in paths:
        # if name == '201809041126441' :
        #     print('a')
        if name in i:
            temp = i.split(name)[-1][0]
            if  temp not in ['0','1','2','3','4','5','6','7','8','9']:
                label_path = i
                j = j+1
                break
    return j,label_path


def set_xml_data(path):
    xml_path = glob(os.path.join(path, '*.xml'))
    img_name = []           #所有图片名称
    all_img_label  = []             #所有图片的标签

    j = 0
    for big_img_xml in tqdm(xml_path):            #所有xml
        p = ParseXml(big_img_xml)
        img_name_, class_list, bbox_list,jpg_or_JPG = p.get_bbox_class()
        big_img_path = os.path.join('/home/wzh/data/img',big_img_xml.split('/')[-1].replace('xml',jpg_or_JPG))
        all_bbox_img = Img_ALL_BBox(cv2.imread(big_img_path))           #一张图片中的所有box
        all_bbox_img.img_path = big_img_path
        img_name.append(img_name_)
        j,path = find_labels(img_name_,j)                 #找到对应的切分图片的地址



        if path != '':
            small_img_xml = sorted(glob(os.path.join(path,'outputs/*')))            #得到一张图片所有的xml
            for i,classes in enumerate(class_list):                 #一张图片中每一个box
                try:
                    small_img_xml_path = '/'.join(small_img_xml[0].split('/')[0:-1])+'/'+str(i).zfill(5)+'.xml'
                except:
                    print(path)
                if os.path.exists(small_img_xml_path):
                     try:
                         label,name = from_xml_read_label.read_label(small_img_xml_path)
                     except:
                         print(img_name_)
                     bbox = Bbox(bbox_list[i],label,classes)
                     if classes ==1:
                         all_bbox_img.print_word.append(bbox)
                     else:
                         all_bbox_img.hand_word.append(bbox)


            all_img_label.append(all_bbox_img)

    return all_img_label
       


    # paths = list(set(paths))
    # paths2 = sorted(glob('/home/wzh/ocr_标注数据集转录交付数据/*'))
    # a = [i for i in paths2 if i  not in paths]

    # print(a)
    # print(j)

    #     for i,type in enumerate(class_list):
    #         bbox =  Bbox(bbox_list[i],)
    #         if type == 1:
    #             pt_wd_.append(bbox_list[i])
    #         else:
    #             hd_wd_.append(bbox_list[i])
    #     pt_wd.append(pt_wd_)
    #     hd_wd.append(hd_wd_)
    # return pt_wd,hd_wd,img_name

# def list_pt_wd(pt_wds):
#     list_all = []
#     for pt_wd in pt_wds:
#         if len()
        
def draw_bbox(bbox,img,color):
    if len(bbox) == 4:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    else:

        cv2.line(img, (bbox[0], bbox[1]),
                 (bbox[2], bbox[3]), color, 2)
        cv2.line(img, (bbox[2], bbox[3]),
                 (bbox[4], bbox[5]), color, 2)
        cv2.line(img, (bbox[4], bbox[5]),
                 (bbox[6], bbox[7]), color, 2)
        cv2.line(img, (bbox[6], bbox[7]),
                 (bbox[0], bbox[1]), color, 2)








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



def to_int(a):
    x,y = a
    return (int(x),int(y))

def in_same_line(print_bbox,hand_bbox):
    if len(print_bbox) == 4:
        centre_print_bbox = (print_bbox[1] + print_bbox[3]) / 2
    else:
        centre_print_bbox =  (print_bbox[3] + print_bbox[5]) / 2

    if len(hand_bbox) == 4:
        if centre_print_bbox>hand_bbox[1] and centre_print_bbox<hand_bbox[3]:
            return 'in'
        else:
            return 'out'
    else:
        if centre_print_bbox > hand_bbox[1] and centre_print_bbox < hand_bbox[7]:
            return 'in'
        else:
            return 'out'

def get_box_centre(box):
    if len(box) == 4:
        centre = ((box[0]+box[2])/2,(box[1]+box[3])/2)
    else:
        centre = ((box[0]+box[4])/2,(box[1]+box[5])/2)
    return to_int(centre)




def get_pair_by_distance(all_bbox):
    # all_bbox = Img_ALL_BBox()

    print_hand = {}
    hand_print = {}
    for i,print_word in enumerate(all_bbox.print_word):
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(all_bbox.hand_word):
            distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
            if min_distance>distance:
                pair = j
                min_distance = distance

        try:
            if in_same_line(print_word.bbox,all_bbox.hand_word[pair].bbox) == 'in':
                print_hand[i] = pair
                if hand_print.get(pair):
                    hand_print[pair].append(i)
                else:
                    hand_print[pair] = [i]
        except:
            pass


    for key in hand_print:
        if len(hand_print[key])>1:
            min_distance = 9999
            min_value = -1
            for print in hand_print[key]:
                print_word = all_bbox.print_word[print]
                hand_word = all_bbox.hand_word[key]
                distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
                if min_distance>distance:
                    min_distance = distance
                    print_hand.pop(min_value,'none')
                    min_value = print
                else:
                    print_hand.pop(print)


    return print_hand,hand_print

def get_centre(box):
    if len(box) == 4:
        point = (int((box[2]+box[0])/2), int((box[1] + box[3]) / 2))
    else:
        point = (int((box[0] + box[4]) / 2), int((box[1] + box[5]) / 2))

    return point







def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))



if __name__ == '__main__':


    xml_path = '/home/wzh/data/xml'
    all_img = set_xml_data(xml_path)


    for img_result in tqdm(all_img):
        print_hand, hand_print = get_pair_by_distance(img_result)
        img_result.pair = print_hand
        # img = cv2.imread(img_result.img_path)
        # path = img_result.img_path.replace('data/img','pair_version1')
        # for pair in print_hand:
        #     bbox_print = img_result.print_word[pair]
        #     bbox_hand = img_result.hand_word[print_hand[pair]]
        #     draw_bbox(bbox_print.bbox, img, (0, 0, 255))
        #     draw_bbox(bbox_hand.bbox, img, (255, 0, 0))
        #     cv2.line(img, get_box_centre(bbox_print.bbox), get_box_centre(bbox_hand.bbox), (0, 0, 0), 2)
        #
        # cv2.imwrite(path,img)
    for i,img_result in tqdm(enumerate(all_img)):
        img_result.cut_biger_img('/home/wzh/ocr_train_bigger/',i)

    # img_result = random.sample(all_img,1)[0]
    # img_result.img = cv2.imread(img_result.img_path)





    
    # cv2.imshow('a',img)
    # cv2.waitKey()

    # img_name = img_name[0].replace('xml', 'JPG')
    # img_path = '/home/wzh/ocr-demo/pic/'
    # if not os.path.exists(img_path):
    #     os.mkdir(img_path)
    # img = cv2.imread(img_path + img_name)


    # for i, point_i in enumerate(pt_wd[0]):
    #     for j, point_j in enumerate(hd_wd[0]):
    #         pt_wd_point, hd_wd_ponit = bbbox_to_distance(point_i, point_j)
    #         cv2.circle(img, pt_wd_point, 10, (255, 0, 0))
    #         cv2.circle(img, hd_wd_ponit, 10, (255, 0, 0))



    # pair = get_pair(pt_wd[0], hd_wd[0])
    # write_line(img,pt_wd[0], hd_wd[0],pair)
    # cv2.imwrite("1.jpg",img)
    # cv2.waitKey()