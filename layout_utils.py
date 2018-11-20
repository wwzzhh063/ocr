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

def no_chinese(check_str):
    '''
    检查字符串中是否含有中文
    :param check_str:   被检查的字符串
    :return:    True or False
    '''

    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    return True




def get_distance(point1, point2):
    '''
    得到两点间的距离
    :param point1:  点1
    :param point1:  点2
    :return:    返回距离
    '''
    points = zip(point1, point2)
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

        #包含关系
        if (print_bbox[1] >= hand_bbox[1] and print_bbox[3]<= hand_bbox[3]) or (hand_bbox[1]>= print_bbox[1] and hand_bbox[3]<=print_bbox[3]):
            return 'in'

        # 打印体中心在手写体范围内
        elif centre_print_bbox>hand_bbox[1] and centre_print_bbox<hand_bbox[3]:
            return 'in'

        #行iou>0.9
        elif row_iou(print_bbox,hand_bbox)>0.9:
            return 'in'
        else:
            return 'out'
    else:
        if centre_print_bbox > hand_bbox[1] and centre_print_bbox < hand_bbox[7]:
            return 'in'
        else:
            return 'out'




def row_iou(row1,row2):
    max_top = max(row1[1],row2[1])
    min_bottom = min(row1[3],row2[3])

    if max_top>=min_bottom:
        return 0
    else:
        return (min_bottom-max_top)/min(row1[3]-row1[1],row2[3]-row2[1])

def column_iou(column1,column2,type = 'min'):
    max_left = max(column1[0],column2[0])
    min_right = min(column1[2],column2[2])

    if max_left>=min_right:
        return 0
    else:               #分为与大的算iou,小的算iou,普通的iou算法
        if type =='min':
            return (min_right-max_left)/min(column1[2]-column1[0],column2[2]-column2[0])
        elif type == 'max':
            return (min_right-max_left)/max(column1[2]-column1[0],column2[2]-column2[0])
        else:
            pass



def row_get_pair(print_cell_word_all,hand_word_all,min_value = 3.5):


    def bbbox_to_distance(point_i, point_j):
        '''
        根据两个bbox计算他们之间的距离
        计算的是box2的left-box1的right再取绝对值
        适用于横向匹配
        :param point_i:
        :param point_j:
        :return:
        '''
        if len(point_i) == 4:
            print_cell_word_point = (point_i[2], (point_i[1] + point_i[3]) / 2)
        else:
            print_cell_word_point = ((point_i[2] + point_i[4]) / 2, (point_i[3] + point_i[5]) / 2)

        if len(point_j) == 4:
            hand_word_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
        else:
            hand_word_ponit = ((point_j[0] + point_j[6]) / 2, (point_j[1] + point_j[7]) / 2)

        distences = get_distance(hand_word_ponit, print_cell_word_point)

        return distences


    def best_pair(box1,box2):
        '''
        如果满足下面条件,则直接判断这两个框为匹配框
        :param box1:
        :param box2:
        :return:
        '''
        if row_iou(box1,box2)>0.9 and column_iou(box1,box2)>0.3:
            return True

    def row_pair_cond(box1,box2,min_value):
        '''
        看两个配对框是否满足配对条件
        :param box1:框1,这个版本为打印框或者merge之后的框

        :param box2:框2,这个版本为手写框,或者带括号等式的打印框
        :return:
        '''
        same_line = in_same_line(box1.bbox,box2.bbox)=='in'         #在同一行

        pair_distance1 = box2.left - box1.right < (box1.right - box1.left)/ min_value

        pair_distance2 = abs(box2.left - box1.right) < (box1.right - box1.left)

        c_iou = column_iou(box1.bbox,box2.bbox)<0.9              #过滤草稿,对应conflunece上免的badcase1

        return same_line and pair_distance1 and pair_distance1 and c_iou


    print_cell_hand = {}
    hand_print_cell = {}
    for i,print_cell_word in enumerate(print_cell_word_all):         #手写到打印匹配一遍
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(hand_word_all):               #算距离
            distance = bbbox_to_distance(print_cell_word.bbox,hand_word.bbox)
            if min_distance>distance :                        #应该是这么加,但是有bug           and row_pair_cond(print_cell_word, hand_word_all[j],min_value) :
                pair = j
                min_distance = distance

            #手写打印重合
            if best_pair(print_cell_word.bbox,hand_word.bbox):
                pair = j
                break
        try:
            if row_pair_cond(print_cell_word, hand_word_all[pair],min_value):
                print_cell_hand[i] = pair
                if hand_print_cell.get(pair):
                    hand_print_cell[pair].append(i)
                else:
                    hand_print_cell[pair] = [i]
        except:
            print('a')


    for key in hand_print_cell:                      #打印到手写再匹配一遍
        if len(hand_print_cell[key])>1:
            min_distance = 9999
            min_value = -1
            for print_cell in hand_print_cell[key]:
                print_cell_word = print_cell_word_all[print_cell]
                hand_word = hand_word_all[key]
                distance = bbbox_to_distance(print_cell_word.bbox,hand_word.bbox)
                if min_distance>distance:
                    min_distance = distance
                    print_cell_hand.pop(min_value,'none')
                    min_value = print_cell
                else:
                    print_cell_hand.pop(print_cell)


    return print_cell_hand


def column_get_pair(boxes):



    def bbbox_to_distance(point_i, point_j):
        if len(point_i) == 4:
            print_cell_word_point = (point_i[0], (point_i[1] + point_i[3]) / 2)
        else:
            pass

        if len(point_j) == 4:
            hand_word_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
        else:
            pass

        distences = get_distance(hand_word_ponit, print_cell_word_point)

        return distences


    def column_pair_cond(box_top,box_bottom,distance,min_distance):
        #要有列iou
        c_iou = column_iou(box_top.bbox, box_bottom.bbox) > 0.1

        #垂直方向的距离不能过远
        column_distance1 = (distance < (box_top.bottom - box_top.top) * 4 or distance < (box_bottom.bottom - box_bottom.top) * 4)

        #垂直方向的距离很小
        column_distance2 = (distance < min_distance and distance < (box_top.bottom - box_top.top) * 2)

        #距离更小
        minimum = distance<min_distance

        return minimum and c_iou and column_distance1 or column_distance2




    box_top_to_bottom = {}
    box_bottom_to_top = {}

    for i,box_top in enumerate(boxes):
        min_distance = 9999
        pair = -1

        for j,box_bottom in enumerate(boxes):

            if box_top is box_bottom:
                continue
            elif box_top.top > box_bottom.top:
                continue
            else:

                distance = bbbox_to_distance(box_top.bbox,box_bottom.bbox)

                if  column_pair_cond(box_top,box_bottom,distance,min_distance):

                    min_distance = distance
                    pair = j

        box_top_to_bottom[i] = pair



        if box_bottom_to_top.get(pair):
            box_bottom_to_top[pair].append(i)
        else:
            box_bottom_to_top[pair] = [i]

    return box_top_to_bottom,box_bottom_to_top


if __name__ == '__main__':
    get_distance((1,2),(3,4))