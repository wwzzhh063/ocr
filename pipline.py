import utils
import sys
import tensorflow as tf
from model import CTC_Model
from config import Config as config
import cv2

from PIL import Image,ImageDraw,ImageFont
sys.path.append('Arithmetic_Func_detection_for_CTPN_v2')
from Arithmetic_Func_detection_for_CTPN_v2.ctpn import run

import math
import argparse
import time
from glob import glob
from tqdm import tqdm
import os
import re
from inference import set_xml_data
import Levenshtein
from utils import bbbox_to_distance,in_same_line


def row_iou(row1,row2):
    max_top = max(row1[0],row2[0])
    min_bottom = min(row1[1],row2[1])

    if max_top>=min_bottom:
        return 0
    else:
        return (min_bottom-max_top)/(row1[1]-row1[0]+row2[1]-row2[0]+max_top-min_bottom)

def column_iou(column1,column2):
    max_left = max(column1[0],column2[0])
    min_right = min(column1[2],column2[2])

    if max_left>=min_right:
        return 0
    else:
        try:
            return (min_right-max_left)/(column1[2]-column1[0]+column2[2]-column2[0]-min_right+max_left)
        except:
            print('a')

def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))




class Result(object):
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


    def set_result(self,result):
        self.result = result



def row_get_pair_by_distance(print_cell_word_all,hand_word_all,min_value = 3.5):

    def bbbox_to_distance(point_i, point_j):
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


    print_cell_hand = {}
    hand_print_cell = {}
    for i,print_cell_word in enumerate(print_cell_word_all):         #手写到打印匹配一遍
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(hand_word_all):               #算距离
            distance = bbbox_to_distance(print_cell_word.bbox,hand_word.bbox)
            if min_distance>distance :
                pair = j
                min_distance = distance

        try:
            if in_same_line(print_cell_word.bbox,hand_word_all[pair].bbox) == 'in' and min_distance<(print_cell_word.bbox[2]-print_cell_word.bbox[0])/min_value \
                    and column_iou(print_cell_word.bbox,hand_word_all[pair].bbox)<0.1:  #算是否在一行
                print_cell_hand[i] = pair
                if hand_print_cell.get(pair):
                    hand_print_cell[pair].append(i)
                else:
                    hand_print_cell[pair] = [i]
        except:
            pass


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




def column_get_pair_by_distance(boxes):



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



    box_top_to_bottom = {}
    box_bottom_to_top = {}

    for i,box_top in enumerate(boxes):
        min_distance = 9999
        pair = -1
        min_top = 9999
        for j,box_bottom in enumerate(boxes):
            # if i == 12 and j == 22:
            #     print('a')

            if box_top is box_bottom:
                continue
            else:
                distnace = bbbox_to_distance(box_top.bbox,box_bottom.bbox)
                top = box_bottom.top
                # distnace = box_bottom.top - box_top.top
                if box_top.top>box_bottom.top:
                    continue
                if  distnace<min_distance and column_iou(box_top.bbox,box_bottom.bbox)>0.1 \
                        and ((distnace < (box_top.bottom-box_top.top)*4 or distnace < (box_bottom.bottom-box_bottom.top)*4))  \
                        or (distnace<min_distance and distnace < (box_top.bottom-box_top.top)*2):
                    min_distance = distnace
                    min_top = top
                    pair = j
        box_top_to_bottom[i] = pair

        if box_bottom_to_top.get(pair):
            box_bottom_to_top[pair].append(i)
        else:
            box_bottom_to_top[pair] = [i]

    return box_top_to_bottom,box_bottom_to_top


class All_Result(object):
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
            result = Result(box, self.img, self.types[i])
            if result.type == 'print':
                self.print_word.append(result)
            else:
                self.hand_word.append(result)

    def create_input(self):
        imgs = []
        wides = []

        for i,box in enumerate(self.all_after_row_connect):
            result = Result(box.bbox,self.img,self.types[i])
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
            big_img = Result([left, top, right, bottom],self.img,type='merge')
            # if '*' in big_img.label or '~' in big_img.label:
            #     big_img.type = '...'
            merge.append(big_img)

        return box_list1_,box_list2_,merge


    def row_connect(self):
        print_cell_hand = row_get_pair_by_distance(self.print_word, self.hand_word)
        print_cell_residue,hand_residue,merge = self.create_big_img(print_cell_hand, self.print_word, self.hand_word)
        merge_print_cell = row_get_pair_by_distance(merge,print_cell_residue,10)
        merge_residue,print_cell_residue,merge = self.create_big_img(merge_print_cell,merge,print_cell_residue)
        self.row_pairs = merge_residue+merge
        self.hand_after_row_connect = hand_residue
        self.print_after_row_connect = print_cell_residue
        return self.row_pairs


    def column_connect(self):
        self.all_after_row_connect = self.row_pairs+self.hand_after_row_connect+self.print_after_row_connect
        self.column_pairs,_ = column_get_pair_by_distance(self.all_after_row_connect)
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












class Evaluate_Data(object):
    def __init__(self):
        self.equation_all = 0
        self.equation_right = 0
        self.bracket_all = 0
        self.bracket_right = 0
        self.residual_all = 0
        self.residual_right = 0
        self.state_all = 0
        self.state_right = 0
        self.char_acc = 0
        self.recall = 0

        self.char_all = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'+':0,'-':0,'×':0,'÷':0,'=':0,'*':0,'~':0,'(':0,')':0}


        self.char_recall = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'+':0,'-':0,'×':0,'÷':0,'=':0,'*':0,'~':0,'(':0,')':0}

        self.char_right = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'+':0,'-':0,'×':0,'÷':0,'=':0,'*':0,'~':0,'(':0,')':0}

    def compute(self):
        self.all = self.residual_all+self.bracket_all+self.equation_all
        self.right = self.residual_right+self.bracket_right+self.equation_right
        self.all_recall = self.right/self.all
        self.state_recall = self.state_right/self.state_all


        if self.equation_all == 0:
            self.equation_recall = 1
        else:
            self.equation_recall = self.equation_right/self.equation_all
        if self.bracket_all == 0:
            self.bracket_recall = 1
        else:
            self.bracket_recall = self.bracket_right/self.bracket_all
        if self.residual_all == 0:
            self.residual_recall = 1
        else:
            self.residual_recall = self.residual_right/self.residual_all

        self.char_acc = self.char_acc/self.all


        self.evaluate_dict = {'all':self.all_recall,'=':self.equation_recall,'()':self.bracket_recall,'...':self.residual_recall,'state':self.state_recall,'char_acc':
                              self.char_acc,'recall':self.recall}

        for char in self.char_all:
            if self.char_all[char] == 0:
                self.char_recall[char] = 1
            else:
                try:
                    self.char_recall[char] = self.char_right[char]/self.char_all[char]
                except:
                    print('a')







def draw_box(img,all_result,x_pro,y_pro):
    for result in all_result:
        if result.state == 'right':
            rgb = (0,255,0)
            cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                          (int(result.right * x_pro), int(result.bottom * y_pro)), rgb, 4)
    for result in all_result:
        if result.state == 'error':
            rgb = (255,0,0)
            cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                          (int(result.right * x_pro), int(result.bottom * y_pro)), rgb, 4)
    for result in all_result:
        if result.state == 'problem':
            rgb = (0,0,255)
            cv2.rectangle(img,(int(result.left*x_pro),int(result.top*y_pro)),(int(result.right*x_pro),int(result.bottom*y_pro)),rgb,4)


def draw_bbox(bbox,img,x_pro,y_pro,color):
    cv2.rectangle(img, (int(bbox[0] * x_pro), int(bbox[1] * y_pro)),
                  (int(bbox[2] * x_pro), int(bbox[3] * y_pro)), color, 4)

def draw_result(img,all_result,x_pro,y_pro):
    ttfont = ImageFont.truetype('SimSun.ttf',25)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for result in all_result:
        if result.state == 'right':
            draw.text((int(result.left*x_pro),int(result.top*y_pro)),result.output,fill='blue',font=ttfont)
        else:
            draw.text((int(result.left*x_pro),int(result.top*y_pro)),result.output,fill='blue',font=ttfont)
        # cv2.putText(img,result.output,(result.left,result.top),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    return img






def create_sess():
    g1_config = tf.ConfigProto(allow_soft_placement=True)

    g1 = tf.Graph()
    sess1 = tf.Session(config=g1_config, graph=g1)

    with sess1.as_default():
        with g1.as_default():
            net, run_list = run.build_ctpn_model()
            saver = tf.train.Saver()
            saver.restore(sess1,
                          'Arithmetic_Func_detection_for_CTPN_v2/checkpoints/VGGnet_fast_rcnn_iter_65000.ckpt')



    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2)
    with sess2.as_default():
        with g2.as_default():
            inputs = tf.placeholder(tf.float32, [None, 32, None, 1])
            width = tf.placeholder(tf.int32, [None])
            is_training = tf.placeholder(tf.bool)
            model = CTC_Model()
            logits, sequence_length = model.crnn(inputs, width, is_training)

            decoder, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length, merge_repeated=True)
            decoder = decoder[0]

            dense_decoder = tf.sparse_to_dense(sparse_indices=decoder.indices, output_shape=decoder.dense_shape,
                                               sparse_values=decoder.values, default_value=-1)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess2, config.MODEL_SAVE)

            return sess1,sess2,net,run_list,dense_decoder,inputs,width,is_training


def delete_pair_problem_result(label1,label2):
    label = label1+label2

    state, revise_output, output = delete_top_or_bottom(label)
    if state != 'right':
        label = label1[0:len(label1)-1]+label2
        state, revise_output, output = delete_top_or_bottom(label)
    if state != 'right':
        label = label1+label2[1:]
        state, revise_output, output = delete_top_or_bottom(label)
    if state != 'right':
        label = label1[0:len(label1)-1]+label2[1:]
        state, revise_output, output = delete_top_or_bottom(label)
    if state != 'right':
        label = label1+'='+label2[1:]
        state, revise_output, output = delete_top_or_bottom(label)


    return state, revise_output, output


def delete_top_or_bottom(label):
    label_temp = label

    try:
        result = eval_label(label)
    except:
        result = 'problem'

    result_temp = result

    if result != 'right':
        label_temp = label[1:len(label)]
        try:
            result = eval_label(label_temp)
        except:
            pass
    if result != 'right':
        label_temp = label[2:len(label)]
        try:
            result = eval_label(label_temp)
        except:
            pass

    if result != 'right':
        label_temp = label[0:len(label)-1]
        try:
            result = eval_label(label_temp)
        except:
            pass

    if result != 'right':
        label_temp = label[1:len(label) - 1]
        try:
            result = eval_label(label_temp)
        except:
            pass

    if result != 'right':
        label_temp = label[2:len(label)-1]
        try:
            result = eval_label(label_temp)
        except:
            pass

    if result != 'right':                 #可能会把error变为problem
        result = result_temp


    return result,label_temp,label



def pro_problem_to_right(label):
    if len(re.split('[+,-,×,÷,(,)]', label)[0]) > 3:
        label = label[1:]
    num = 0;
    num = correct_problem(label)
    if num > 0:
        return label
    if correct_problem(label[1:]) > num and eval_label(label[1:])!='problem':
        num = correct_problem(label[1:])
        label = label[1:]
    if correct_problem(label[0:len(label) - 1]) > num and eval_label(label[0:len(label) - 1])!='problem':
        num = correct_problem(label[0:len(label) - 1])
        label = label[0:len(label) - 1]
    if correct_problem(label[1:len(label) - 1]) > num and eval_label(label[1:len(label) - 1])!='problem':
        num = correct_problem(label[1:len(label) - 1])
        label = label[1:len(label) - 1]

    return label


def correct_problem(label):
    num = 0
    label_list = label.split('*')
    label = list(set(label_list))
    label.sort(key=label_list.index)
    label = '*'.join(label)
    for i in range(len(label)):
        for j in '1234567890':
            label_ = label[:i]+j+label[i+1:]
            try:
                result = eval_label(label_)
            except:
                result = 'problem'
            if result=='right':
                num = num+1
    return num

def add_bracket(label):                         #加到修改错误项的前面
    if '=' not in label or label=='':
        return label,'problem'
    else:
        left = label.split('=')[0]
        right = label.split('=')[1]

    if right == '' or left == '':
        return label,'problem'

    error_list = []

    if '(' in left and ')' not in left:
        left_num = re.split('[+,-,*,/,(]',left)
        for num in left_num:
            left_temp = left.replace(num,num+')')
            try:
                label_temp = left_temp + '=' + right
                state = eval_label(label_temp)
                if state == 'right':
                    return left_temp, state
                elif state == 'error':
                    error_list.append(label_temp)
            except:
                pass


    elif ')' in left and '(' not in left:
        error_list = []
        left_num = re.split('[+,-,*,/,(]', left)
        for num in left_num:
            left_temp = left.replace(num, '(' + num)
            try:
                label_temp = left_temp+'='+right
                state = eval_label(label_temp)
                if state == 'right':
                    return left_temp,state
                elif state == 'error':
                    error_list.append(left_temp)
            except:
                pass

    else:
        return label,'problem'

    if len(error_list) != 0 :

        num = 0
        label = error_list[0]
        for error in error_list:
            num_temp = correct_problem(error)
            if num_temp > num:
                num = num_temp
                label = error

        return label,'error'

    else:
        return label,'problem'




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


def revise_result(result):
    state, revise_output, output = delete_top_or_bottom(result)
    if state == 'right':
        output = revise_output
        return 'right',revise_output
    if state == 'problem':
        output, state = add_bracket(output)
        if state == 'right':
            return 'right',output
    if state == 'error':
        output = pro_problem_to_right(output)
    return state,output

def pipeline(img,sess1,sess2,net, run_list,dense_decoder,inputs,width,is_training):
    feed_dict, img_resized_shape, im_scales, bbox_scale = run.run_ctpn(img, net)
    out_put = sess1.run(run_list, feed_dict)

    bboxes,types = run.decode_ctpn_output(out_put, im_scales, bbox_scale, img_resized_shape)

    types2 = []
    for i,type in enumerate(types):
        if type == 1:
            types2.append('hand')
        else:
            types2.append('print')


    all_result = All_Result(bboxes,img,types2)
    all_result.row_connect()
    all_result.column_connect()
    all_result.graph_to_forest()
    image,wides = all_result.create_input()

    sentence = sess2.run(dense_decoder, feed_dict={inputs: image, width: wides, is_training: False})

    output = sentence.tolist()

    decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

    output = list(map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))).replace('－','-'), output))

    problems = []

    ave_wide = 0
    ave_high = 0

    for i, result in enumerate(all_result.all_after_row_connect):
        result.output = output[i]
        if result.type == 'merge':
            result.state,result.output = revise_result(result.output)
            all_result.connect_result.append(result)



        elif result.type == 'print':
            if '@' in result.output:
                all_result.fraction.append(result)
            else:
                all_result.vertical.append(result)

    for result in all_result.vertical:
        # if result.output == '350-8×6+5×4':
        #     print('a')
        bottom_list = []
        position = result.position
        forest = all_result.forest_list[position[0]]
        for num in range(position[1]+1, len(forest)):
            bottom = forest[num]
            if bottom.type == 'print' or bottom.type == 'merge':
                break
            else:
                label = result.output.replace('=', '') + '=' + bottom.output.replace('=', '')

                state, output = revise_result(label)
                bottom_list.append(bottom)
                if state == 'right':
                    result.output = output
                    result.state = state
                    all_result.connect_result.append(result)
                    break
        if result.state != 'right':
            bottom_list.append(result)
            result.left = min([bottom.left for bottom in bottom_list])
            result.right =  max([bottom.right for bottom in bottom_list])
            result.top = min([bottom.top for bottom in bottom_list])
            result.bottom = max([bottom.bottom for bottom in bottom_list])
            result.bbox = [result.left,result.top,result.right,result.bottom]
            result.state = 'error'
            result.output = ''
            all_result.connect_result.append(result)



















    #合并有问题的检测框-----------------------------------------------------------------------------------------用于单分类
    # delete = []
    # for i, problem1 in enumerate(temp):
    #     problems_without = temp.copy()
    #     problems_without.pop(i)
    #
    #     for j, problem2 in enumerate(problems_without):
    #         if  min(problem1.bottom, problem2.bottom) > max(problem1.top, problem2.top) \
    #                 and problem2.left >= problem1.right and (problem2.left - problem1.right) < 10/bbox_scale and not set('*——×÷')&set(problem2.output):
    #             state, revise_output, output = delete_pair_problem_result(problem1.output, problem2.output)
    #             box = [min(problem1.left, problem2.left), min(problem1.top, problem2.top),
    #                    max(problem1.right, problem2.right), max(problem1.bottom, problem2.bottom)]
    #             result = Result(box, img)
    #             result.state = state
    #             result.revise_output = revise_output
    #             result.output = output
    #
    #             if result.state == 'right':
    #                 all_result.right_result.append(result)
    #             elif result.state == 'error' :
    #                 all_result.error_result.append(result)
    #
    #
    #             try:
    #                 all_result.problem_result.remove(problem1)
    #                 all_result.problem_result.remove(problem2)
    #             except:
    #                 pass

    # for i,problem in enumerate(all_result.problem_result.copy()):
    #
    #     output,state = add_bracket(problem.output)
    #     if state == 'right':
    #         all_result.problem_result.remove(problem)
    #         problem.revise_output = output
    #         all_result.right_result.append(problem)
    #     elif state == 'error':
    #         all_result.problem_result.remove(problem)
    #         problem.state = state
    #         problem.output = output
    #         all_result.error_result.append(problem)
    # #
    # #
    # #
    # #
    # #
    # all_result.connect_result.extend(all_result.problem_result)
    # all_result.connect_result.extend(all_result.right_result)
    # all_result.connect_result.extend(all_result.error_result)

            # pass


    # for result in all_result.connect_result.copy():
    #     if len(set(result.output)&set('+-×÷()=')) <=1:
    #         all_result.connect_result.remove(result)

    return all_result,bboxes


def area(point1,point2):
    return max((point2[0]-point1[0]),0)*max((point2[1]-point1[1]),0)





def get_iou(box1,box2):
    area1 = area((box1[0],box1[1]),(box1[2],box1[3]))
    area2 = area((box2[0],box2[1]),(box2[2],box2[3]))

    point1 = (max(box1[0],box2[0]),max(box1[1],box2[1]))
    point2 = (min(box1[2],box2[2]),min(box1[3],box2[3]))

    area3 = area(point1,point2)

    return area3/(area1+area2-area3)



def evaluate():

    xml_path = '/home/wzh/data3/xml'
    all_img = set_xml_data(xml_path)



    sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training = create_sess()

    evaluate_data = Evaluate_Data()

    for i,img_result in tqdm(enumerate(all_img)):
        img_result.create_pair()
        img = img_result.img

        result = pipeline(img.copy(), sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training)

        result_pair = {}

        for i,pre_box in enumerate(result.connect_result):
            max_iou = 0
            pair = -1
            for j,true_box in enumerate(img_result.all_box):
                iou = get_iou(pre_box.bbox,true_box.bbox)
                if iou>max_iou:
                    max_iou = iou
                    pair = j

            if max_iou>0.3:
                result_pair[i] = pair




        x_pro = 3024 / img.shape[1]
        y_pro = 4031 / img.shape[0]
        img = cv2.resize(img, (3024, 4032))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = img.copy()





        for pre_num in result_pair:
            pre_box = result.connect_result[pre_num]
            true_box = img_result.all_box[result_pair[pre_num]]

            if true_box.type == '=':
                if true_box.label == pre_box.output:
                    evaluate_data.equation_right = evaluate_data.equation_right+1
                evaluate_data.equation_all = evaluate_data.equation_all+1
            elif true_box.type == '()':
                if true_box.label == pre_box.output:
                    evaluate_data.bracket_right = evaluate_data.bracket_right+1
                evaluate_data.bracket_all = evaluate_data.bracket_all+1
            else:
                if true_box.label == pre_box.output:
                    evaluate_data.residual_right = evaluate_data.residual_right+1
                evaluate_data.residual_all = evaluate_data.residual_all+1

            if true_box.state == pre_box.state:
                evaluate_data.state_right = evaluate_data.state_right+1
            evaluate_data.state_all = evaluate_data.state_all+1

            for char in set(true_box.label):
                evaluate_data.char_all[char] = evaluate_data.char_all[char]+1

            for char in set(true_box.label).intersection(pre_box.output):
                evaluate_data.char_right[char] = evaluate_data.char_right[char]+1


            if true_box.state == pre_box.state and true_box.label!= pre_box.output:
                draw_bbox(true_box.bbox, img, x_pro, y_pro, (255, 255, 255))
                print(pre_box.output)
            else:
                draw_bbox(true_box.bbox, img, x_pro, y_pro, (0, 255, 255))



            evaluate_data.char_acc = evaluate_data.char_acc+1-(Levenshtein.distance(true_box.label,pre_box.output)/len(true_box.label))









            # draw_bbox(pre_box.bbox,img,x_pro, y_pro,(255,0,0))
            # draw_bbox(true_box.bbox,img,x_pro, y_pro,(0,0,255))
        lenght = 0
        if len(img_result.all_box)>0:
            evaluate_data.recall = evaluate_data.recall + (len(result_pair) / len(img_result.all_box))
            lenght = lenght+1




        draw_box(img, result.connect_result, x_pro, y_pro)
        img = draw_result(img, result.connect_result, x_pro, y_pro)
        img.save('result/pipeline'+str(i)+'_.jpg')

        # cv2.imwrite('/home/wzh/test2/'+str(i)+'_.jpg',img)

    evaluate_data.recall = evaluate_data.recall / lenght

    evaluate_data.compute()
    print(evaluate_data.evaluate_dict)
    print(evaluate_data.char_recall)

    log = open('log.txt','a')
    log.writelines(str(evaluate_data.evaluate_dict))
    log.writelines(str(evaluate_data.char_recall))

        # img.save('/home/wzh/test2/'+str(i)+'_.jpg')








if __name__ == '__main__':
    # print(eval_label('长='))
    # evaluate()


#    [2246, 1660, 2662, 1765]
# [2271, 1363, 2710, 1448]
#     a = get_iou([2246, 1660, 2662, 1765],[2271, 1363, 2710, 1448])
#     print(a)





    parser = argparse.ArgumentParser()
    parser.add_argument('path',help='image path',type=str)
    args = parser.parse_args()
    img = cv2.imread(args.path)
    #

    sess1, sess2, net, run_list, dense_decoder,inputs,width,is_training = create_sess()
    result,bboxes = pipeline(img.copy(), sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training)
    x_pro = 3024 / img.shape[1]
    y_pro = 4031 / img.shape[0]
    img = cv2.resize(img, (3024, 4032))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = img.copy()
    draw_box(img, result.connect_result, x_pro, y_pro)
    img = draw_result(img, result.connect_result, x_pro, y_pro)

    for box in bboxes:
        draw_bbox(box,img2, x_pro, y_pro,(0,0,255))
    img.show()
    img.save(args.path.replace('.','_.'))
    cv2.imwrite(args.path.replace('.','_2.'),img2)







    # sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training = create_sess()
    # img_path = glob('/home/wzh/test16/*')
    # for path in tqdm(img_path):
    #     img = cv2.imread(path)
    #     save_path = path.replace('test16','pipline_result/test16_3')
    #
    #
    #     result = pipline(img.copy(),sess1, sess2, net, run_list, dense_decoder,inputs,width,is_training)
    #     x_pro = 3024 / img.shape[1]
    #     y_pro = 4031 / img.shape[0]
    #     img = cv2.resize(img, (3024, 4032))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     # draw_box(img,result.right_error_result,x_pro,y_pro)
    #     # img = draw_result(img,result.right_error_result,x_pro,y_pro)
    #     # draw_box(img,result.problem_result,x_pro,y_pro)
    #     # img = draw_result(img,result.problem_result,x_pro,y_pro)
    #
    #     draw_box(img, result.connect_result, x_pro, y_pro)
    #     img = draw_result(img, result.connect_result, x_pro, y_pro)
    #
    #     # draw_box(img, result.results, x_pro, y_pro)
    #     # img = draw_result(img, result.results, x_pro, y_pro)
    #
    #     img.save(save_path)
    #