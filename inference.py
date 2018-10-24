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
from PIL import Image,ImageDraw,ImageFont


def eval_label(label):
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



class Bbox(object):
    def __init__(self,bbox,label,classes):
        self.bbox = bbox
        self.label = label
        self.classes = classes
        self.state = 'start'
        self.type = '='

        if len(bbox) == 4:
            self.top = bbox[1]
            self.bottom = bbox[3]
            self.left = bbox[0]
            self.right = bbox[2]
        else:
            self.top = min(bbox[1],bbox[3],bbox[5],bbox[7])
            self.bottom = max(bbox[1],bbox[3],bbox[5],bbox[7])
            self.left = min(bbox[0],bbox[6],bbox[2],bbox[4])
            self.right = max(bbox[0],bbox[6],bbox[2],bbox[4])
            self.bbox = [self.left,self.top,self.right,self.bottom]



class Img_ALL_BBox(object):
    def __init__(self,img):
        self.print_word = []
        self.hand_word = []
        self.merge = []
        self.img_path = ''
        self.pair = {}
        self.img = img
        self.not_pair = []
        self.all_box = []
        self.problem_box = []

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
            if 'None' not in label:                     #test----------------------------------------
                cv2.imwrite(path+str(num)+'_'+str(i)+'_'+label+'.jpg',biger_img)

    def create_big_img(self,pair,box_list1,box_list2):        #先合并非括号填词
        box_list1_ = box_list1.copy()
        box_list2_ = box_list2.copy()
        try:
            for i, print_num in enumerate(pair):
                hand_num = pair[print_num]
                print = box_list1_[print_num]
                hand = box_list2_[hand_num]
                box_list1.remove(print)
                box_list2.remove(hand)
                top = min(print.top, hand.top)
                bottom = max(print.bottom, hand.bottom)
                left = min(print.left, hand.left)
                right = max(print.right, hand.right)


                label = print.label + hand.label
                big_img = Bbox([left,top,right,bottom],label,'merge')
                if '*' in big_img.label or '~' in big_img.label:
                    big_img.type = '...'
                self.merge.append(big_img)
        except:
            print('a')

        self.not_pair = box_list1+box_list2


    def create_big_img2(self,pair,box_list1,box_list2):         #合并括号填词的
        box_list1_ = box_list1.copy()
        box_list2_ = box_list2.copy()
        try:
            for i, print_num in enumerate(pair):
                hand_num = pair[print_num]
                print = box_list1_[print_num]
                hand = box_list2_[hand_num]
                box_list1.remove(print)
                box_list2.remove(hand)
                top = min(print.top, hand.top)
                bottom = max(print.bottom, hand.bottom)
                left = min(print.left, hand.left)
                right = max(print.right, hand.right)


                label = print.label + hand.label
                big_img = Bbox([left,top,right,bottom],label,'merge')
                big_img.type = '()'
                self.merge.append(big_img)
        except:
            print('a')

        self.not_pair = box_list2
        self.all_box = self.merge+self.not_pair





    def create_pair(self):
        print_hand, hand_print = get_pair_by_distance(self.print_word,self.hand_word)
        self.pair = print_hand
        self.create_big_img(self.pair,self.print_word,self.hand_word)
        pair,_ = get_pair_by_distance2(self.merge,self.not_pair)
        self.create_big_img2(pair,self.merge,self.not_pair)
        self.all_box = self.all_box+self.print_word+self.hand_word
        # for bbox in self.all_box.copy():
        #     bbox.state =  eval_label(bbox.label)
        #     if bbox.state == 'problem':
        #         self.all_box.remove(bbox)
        #         self.problem_box.append(bbox)
        #
        # for bbox in self.print_word.copy():
        #     bbox.state = eval_label(bbox.label)
        #     if bbox.state != 'problem':
        #         self.all_box.append(bbox)
        #         self.print_word.remove(bbox)
        #
        #
        # for bbox in self.hand_word.copy():
        #     bbox.state = eval_label(bbox.label)
        #     if bbox.state != 'problem':
        #         self.all_box.append(bbox)
        #         self.hand_word.remove(bbox)




def find_labels(name,j):
    name = name.split('.')[0]
    name = name.split('_')[-1]
    path = config.SHIBIE
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
        big_img_path = os.path.join(config.DATA_IMG,big_img_xml.split('/')[-1].replace('xml',jpg_or_JPG))
        all_bbox_img = Img_ALL_BBox(cv2.imread(big_img_path))           #一张图片中的所有box   tetst---------------------------------
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
                     # bbox = Bbox_test(bbox_list[i], label, classes,small_img_xml_path)          #for test!!!!!!!!!!
                     if classes ==1:
                         all_bbox_img.print_word.append(bbox)
                     else:
                         all_bbox_img.hand_word.append(bbox)


            all_img_label.append(all_bbox_img)

    return all_img_label
       



        
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




def get_pair_by_distance(print_word_all,hand_word_all):
    # all_bbox = Img_ALL_BBox()

    print_hand = {}
    hand_print = {}
    for i,print_word in enumerate(print_word_all):         #手写到打印匹配一遍
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(hand_word_all):               #算距离
            distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
            if min_distance>distance:
                pair = j
                min_distance = distance

        try:
            if in_same_line(print_word.bbox,hand_word_all[pair].bbox) == 'in' and min_distance<(print_word.bbox[2]-print_word.bbox[0])/2:  #算是否在一行
                print_hand[i] = pair
                if hand_print.get(pair):
                    hand_print[pair].append(i)
                else:
                    hand_print[pair] = [i]
        except:
            pass


    for key in hand_print:                      #打印到手写再匹配一遍
        if len(hand_print[key])>1:
            min_distance = 9999
            min_value = -1
            for print in hand_print[key]:
                print_word = print_word_all[print]
                hand_word = hand_word_all[key]
                distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
                if min_distance>distance:
                    min_distance = distance
                    print_hand.pop(min_value,'none')
                    min_value = print
                else:
                    print_hand.pop(print)


    return print_hand,hand_print

def get_pair_by_distance2(print_word_all,hand_word_all):
    # all_bbox = Img_ALL_BBox()

    print_hand = {}
    hand_print = {}
    for i,print_word in enumerate(print_word_all):         #手写到打印匹配一遍
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(hand_word_all):               #算距离
            distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
            if min_distance>distance:
                pair = j
                min_distance = distance

        try:
            if in_same_line(print_word.bbox,hand_word_all[pair].bbox) == 'in' and min_distance<(print_word.bbox[2]-print_word.bbox[0])/3 and eval_label(print_word.label)=='problem':  #算是否在一行
                print_hand[i] = pair
                if hand_print.get(pair):
                    hand_print[pair].append(i)
                else:
                    hand_print[pair] = [i]
        except:
            pass


    for key in hand_print:                      #打印到手写再匹配一遍
        if len(hand_print[key])>1:
            min_distance = 9999
            min_value = -1
            for print in hand_print[key]:
                print_word = print_word_all[print]
                hand_word = hand_word_all[key]
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

def get_left(box):
    if len(box) == 4:
        hand_word_ponit = (box[0], (box[1] + box[3]) / 2)
    else:
        hand_word_ponit = ((box[0] + box[6]) / 2, (box[1] + box[7]) / 2)

    return to_int(hand_word_ponit)

def get_right(box):
    if len(box) == 4:
        print_word_point = (box[2], (box[1] + box[3]) / 2)
    else:
        print_word_point = ((box[2] + box[4]) / 2, (box[3] + box[5]) / 2)

    return to_int(print_word_point)

def draw_result(img,all_result,x_pro=1,y_pro=1):
    ttfont = ImageFont.truetype('SimSun.ttf',25)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for result in all_result:
        if result.state == 'right':
            draw.text((int(result.left*x_pro),int(result.top*y_pro)),result.revise_output,fill='blue',font=ttfont)
        else:
            draw.text((int(result.left*x_pro),int(result.top*y_pro)),result.output,fill='blue',font=ttfont)
        # cv2.putText(img,result.output,(result.left,result.top),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    return img




if __name__ == '__main__':


    xml_path = config.DATA_XML
    all_img = set_xml_data(xml_path)


    for i,img_result in tqdm(enumerate(all_img)):
        print_hand, hand_print = get_pair_by_distance(img_result.print_word,img_result.hand_word)
        img_result.pair = print_hand
        img = cv2.imread(img_result.img_path)
        resize_list = []

        if img.shape[0]>3900:
            resize_list.append((1920/img.shape[0],1080/img.shape[1]))
            resize_list.append((854/img.shape[0],640/img.shape[0]))
        elif img.shape[0]>1800:
            resize_list.append((854/img.shape[0],640/img.shape[0]))





        # for pair in print_hand:
        #     bbox_print = img_result.print_word[pair]
        #     bbox_hand = img_result.hand_word[print_hand[pair]]

            # cv2.line(img, get_box_centre(bbox_print.bbox), get_box_centre(bbox_hand.bbox), (0, 0, 0), 2)

            # cv2.line(img, get_right(bbox_print.bbox), get_left(bbox_hand.bbox), (0, 0, 0), 2)



        # for print in img_result.print_word:
        #     draw_bbox(print.bbox, img, (0, 0, 255))
        #
        # for hand in img_result.hand_word:
        #     draw_bbox(hand.bbox, img,(255, 0, 0))


        img_result.create_pair()

        save_path = config.CLEAN_DATA
        for bbox in img_result.all_box:
            cut_img_list = []
            cut_img = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
            cut_img_list.append(cut_img)
            row_temp = bbox.right-bbox.left
            column_temp = bbox.bottom-bbox.top
            label = bbox.label
            if len(label)>10:
                cut_img_list.append(img[bbox.top - int(column_temp / 7):bbox.bottom + 1 + int(column_temp / 7),
                                    bbox.left - int(row_temp / 5):bbox.right + 1 + int(row_temp / 5)])
            else:
                cut_img_list.append(img[bbox.top - int(column_temp / 7):bbox.bottom + 1 + int(column_temp / 7),
                                    bbox.left - int(row_temp / 5):bbox.right + 1 + int(row_temp / 5)])
                cut_img_list.append(img[bbox.top - int(column_temp / 7):bbox.bottom + 1 + int(column_temp / 7),
                                    bbox.left - int(row_temp / 3):bbox.right + 1 + int(row_temp / 3)])
            if config.ENHANCE:
                for x,cut_img in enumerate(cut_img_list):
                    if len(resize_list) > 0:
                        cut_path = os.path.join(save_path,  '1_'+str(i) +'_'+ str(x)+'_' + '0' + '_' + label + '.jpg')
                        cv2.imwrite(cut_path, cut_img)
                        for j, resize in enumerate(resize_list):
                            try:
                                resize_img = cv2.resize(cut_img, None, fx=resize[1], fy=resize[0])
                                cut_path = os.path.join(save_path,  '1_'+str(i)+'_'+str(x) + '_' + str(j + 1) + '_' + label + '.jpg')
                                cv2.imwrite(cut_path, resize_img)
                            except:
                                pass

                    else:
                        cut_path = os.path.join(save_path, '1_'+ str(i) + '_'+str(x)+'_' + label + '.jpg')
                        cv2.imwrite(cut_path, cut_img)
            else:
                if len(resize_list)>0:
                    cut_path = os.path.join(save_path,  '1_'+str(i) + '_' + '0' + '_' + label + '.jpg')
                    cv2.imwrite(cut_path, cut_img)
                    for j,resize in enumerate(resize_list):
                        resize_img = cv2.resize(cut_img,None,fx=resize[1],fy=resize[0])
                        cut_path = os.path.join(save_path, '1_'+str(i)+'_'+str(j+1)+'_'+label+'.jpg')
                        cv2.imwrite(cut_path,resize_img)

                else:
                    cut_path = os.path.join(save_path,  '1_'+str(i) + '_' + label + '.jpg')
                    cv2.imwrite(cut_path, cut_img)



        #验证------------------------------------------------------------------------------------------------------------
        # img2 = cv2.imread(img_result.img_path)
        # path = img_result.img_path.replace('data/img', 'pair/version3')
        # path2 = img_result.img_path.replace('data/img', 'pair/version4')
        # for bbox in img_result.all_box:
        #     draw_bbox(bbox.bbox,img,(0, 0, 255))
        #
        #
        # for bbox in img_result.print_word:
        #     draw_bbox(bbox.bbox,img2,(255, 0, 0))
        #     cv2.putText(img2,bbox.label,(bbox.bbox[0],bbox.bbox[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        #
        #
        # for bbox in img_result.hand_word:
        #     draw_bbox(bbox.bbox, img2, (0, 255, 0))
        #     cv2.putText(img2, bbox.label, (bbox.bbox[0], bbox.bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0,0 ), 1)
        #
        # for bbox in img_result.problem_box:
        #     draw_bbox(bbox.bbox, img2, (0, 0, 255))
        #     cv2.putText(img2, bbox.label, (bbox.bbox[0], bbox.bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        # for print in img_result.not_pair:
        #     draw_bbox(print.bbox, img, (255, 0, 0))


        # cv2.imwrite(path,img)
        # cv2.imwrite(path2, img2)

