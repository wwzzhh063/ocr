import utils
import sys
import tensorflow as tf
from model import CTC_Model
from config import Config as config
import cv2

from PIL import Image,ImageDraw,ImageFont
sys.path.append('Arithmetic_Func_detection_for_CTPN_v2_5')
from Arithmetic_Func_detection_for_CTPN_v2_5.ctpn import run

import math
import argparse
import time
from glob import glob
from tqdm import tqdm
import os
import re
from inference import set_xml_data
import Levenshtein
from layout_utils import row_get_pair,column_get_pair
from utils import draw_pair
from math import log
import numpy as np






class Box_Cell(object):
    def __init__(self,bbox,img,type=''):
        self.top = bbox[1]
        self.bottom = bbox[3]
        self.left = bbox[0]
        self.right = bbox[2]
        self.bbox = bbox
        self.img = img[self.top:self.bottom+1,self.left:self.right+1,...]
        self.normal_img = utils.image_normal(self.img.copy())
        self.img_wide = self.normal_img.shape[1]
        self.output = ''
        self.state = ''
        self.type = type
        self.centre = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
        self.position = (-1,-1)
        self.backup_output = []


    def set_result(self,result):
        self.result = result


class Page(object):
    def __init__(self,bboxes,img,types):
        self.bboxes = bboxes
        self.types = types
        self.img = img
        self.results = []
        self.connect_result = []
        self.max_wide = 0
        self.print_word = []
        self.hand_word = []
        self.equation = []
        self.fraction = []
        self.vertical = []

        for i, box in enumerate(self.bboxes):
            result = Box_Cell(box, self.img, self.types[i])
            if result.type == 'print':
                self.print_word.append(result)
            else:
                self.hand_word.append(result)

    def create_input(self):
        imgs = []
        wides = []

        for i,box in enumerate(self.all_after_row_connect):
            result = Box_Cell(box.bbox,self.img,self.types[i])
            if result.img_wide>self.max_wide:
                self.max_wide = result.img_wide
            wides.append(result.img_wide)
            imgs.append(result.normal_img)
            self.results.append(result)

        inputs,wides = utils.create_input(imgs,self.max_wide,wides)

        return inputs,wides


    def create_big_img(self,pair,box_list1,box_list2):        #先合并非括号填词
        box_list1_ = box_list1.copy()
        box_list2_ = box_list2.copy()
        merge = []
        for i, print_cell_num in enumerate(pair):
            hand_num = pair[print_cell_num]
            print_cell = box_list1[print_cell_num]
            hand = box_list2[hand_num]
            box_list1_.remove(print_cell)
            box_list2_.remove(hand)
            top = min(print_cell.top, hand.top)
            bottom = max(print_cell.bottom, hand.bottom)
            left = min(print_cell.left, hand.left)
            right = max(print_cell.right, hand.right)

            # label = print_cell.label + hand.label
            big_img = Box_Cell([left, top, right, bottom],self.img,type='merge')
            # if '*' in big_img.label or '~' in big_img.label:
            #     big_img.type = '...'
            merge.append(big_img)

        return box_list1_,box_list2_,merge


    def row_connect(self):
        print_cell_hand = row_get_pair(self.print_word, self.hand_word)
        print_cell_residue,hand_residue,merge = self.create_big_img(print_cell_hand, self.print_word, self.hand_word)
        merge_print_cell = row_get_pair(merge,print_cell_residue,10)
        merge_residue,print_cell_residue,merge = self.create_big_img(merge_print_cell,merge,print_cell_residue)
        self.row_pairs = merge_residue+merge
        self.hand_after_row_connect = hand_residue
        self.print_after_row_connect = print_cell_residue
        return self.row_pairs


    def row_connect_test(self):
        print_cell_hand = row_get_pair(self.print_word, self.hand_word)
        draw_pair(print_cell_hand,self.print_word,self.hand_word,self.img,(0,0,255))
        print_cell_residue,hand_residue,merge = self.create_big_img(print_cell_hand, self.print_word, self.hand_word)
        merge_print_cell = row_get_pair(merge,print_cell_residue,10)
        draw_pair(merge_print_cell,merge, print_cell_residue, self.img, (0, 255, 0))
        merge_residue,print_cell_residue,merge = self.create_big_img(merge_print_cell,merge,print_cell_residue)
        self.row_pairs = merge_residue+merge
        self.hand_after_row_connect = hand_residue
        self.print_after_row_connect = print_cell_residue
        return self.row_pairs


    def column_connect(self):
        self.all_after_row_connect = self.row_pairs+self.hand_after_row_connect+self.print_after_row_connect
        self.column_pairs,_ = column_get_pair(self.all_after_row_connect)
        return self.column_pairs


    def graph_to_forest(self):
        forest_num_list = []
        forest_cell_list = []
        for pair in self.column_pairs:
            top = pair
            bottom = self.column_pairs[pair]
            if len(forest_num_list) == 0:
                if bottom == -1:
                    forest_num_list.append([top])
                else:
                    forest_num_list.append([top,bottom])
            else:
                top_forest = []
                bottom_forest = []
                for i,forest in enumerate(forest_num_list):
                    if top in forest:
                        top_forest = forest
                    if bottom in forest:
                        bottom_forest = forest
                    if top_forest and bottom_forest:
                        break

                if top_forest and bottom_forest and top_forest is not bottom_forest:
                    top_forest.extend(bottom_forest)
                    forest_num_list.remove(bottom_forest)
                elif top_forest:
                    if bottom!=-1:
                        top_forest.append(bottom)
                elif bottom_forest:
                    bottom_forest.append(top)
                else:
                    if bottom == -1:
                        forest_num_list.append([top])
                    else:
                        forest_num_list.append([top, bottom])


        for forest in forest_num_list:
            cell_forest = []
            for num in forest:
                cell_forest.append(self.all_after_row_connect[num])
            forest_cell_list.append(cell_forest)

        def forest_sort(node):
            return node.top


        for i,forest in enumerate(forest_cell_list):
            forest.sort(key=forest_sort)
            for j,node in enumerate(forest):
                node.position = (i,j)


        self.forest_list = forest_cell_list