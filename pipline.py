import utils
import sys
import tensorflow as tf
from model import CTC_Model
from config import Config as config
import cv2

from PIL import Image,ImageDraw,ImageFont
sys.path.append('Arithmetic_Func_detection_for_CTPN_v2_5')

from Arithmetic_Func_detection_for_CTPN_v2_5.ctpn import run
# sys.path.append('Arithmetic_Func_detection_for_CTPN_v3')
# from Arithmetic_Func_detection_for_CTPN_v3.ctpn import run

import math
import argparse
import time
from glob import glob
from tqdm import tqdm
import os
import re
from inference import set_xml_data
import Levenshtein
from layout_utils import row_get_pair,column_get_pair,column_iou
from utils import draw_pair,eval_label,draw_bboxes,draw_result
from math import log
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='1'

def beam_search_decoder(data, k = 10):

   sequences = [[list(), 0.0]]

   # walk over each step in sequence

   for row in data:

       all_candidates = list()

       # expand each current candidate

       for i in range(len(sequences)):

           seq, score = sequences[i]

           for j in range(len(row)):
               seq_ = seq.copy()
               if seq and seq[-1] == j and j!=20 and seq[-1]!=20:
                   if seq[-1] == 20:
                       seq_.remove(20)
                   candidate = [seq_, score + (-log(row[j]))]
               else:
                   if seq and seq[-1] == 20:
                       seq_.remove(20)
                   candidate = [seq_ + [j], score + (-log(row[j]))]



               all_candidates.append(candidate)

       # order all candidates by score

       ordered = sorted(all_candidates, key=lambda tup:tup[1])

       # select k best

       # sequences = ordered[:k]

       sequence_list = []
       sequence_temp = []
       for i,sequence in enumerate(ordered):
           if i == 0:
               sequence_list.append(sequence)
               sequence_temp.append(sequence[0])
           else:
               if sequence[0] not in sequence_temp:
                   sequence_list.append(sequence)
                   sequence_temp.append(sequence[0])
           if len(sequence_list) == k:
               break
       sequences = sequence_list

   return sequences


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
        self.bracket_before_merge = []          #括号类的题在合并前的结果 todo

        '''
        当发生如下情况时我们保存合并前的检测结果
        1.该算式可能为竖式
        2.打印框和手写框的尺度相差较大    TODO
        '''
        self.equation_before_merge = []         #竖式类型的题在合并前的结果
        self.output = ''
        self.state = ''
        self.type = type
        self.centre = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
        self.position = (-1,-1)
        self.backup_output = []



    def set_result(self,result):
        self.result = result


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
        self.row_pairs = []
        self.other_result = []


        for i, box in enumerate(self.bboxes):
            result = Result(box, self.img, self.types[i])
            if result.type == 'print':
                self.print_word.append(result)
            else:
                self.hand_word.append(result)

        self.all_after_row_connect = self.print_word+self.hand_word



    def create_input(self):
        imgs = []
        wides = []

        def add_img(result):
            if result.img_wide>self.max_wide:
                self.max_wide = result.img_wide
            wides.append(result.img_wide)
            imgs.append(result.normal_img)


        for i,result in enumerate(self.all_after_row_connect):
            add_img(result)

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

            big_img = Result([left, top, right, bottom],self.img,type='merge')

            big_img.equation_before_merge.append(print_cell)
            big_img.equation_before_merge.append(hand)


            merge.append(big_img)

        return box_list1_,box_list2_,merge


    def row_connect(self):
        print_cell_hand = row_get_pair(self.print_word, self.hand_word)
        print_cell_residue,hand_residue,merge = self.create_big_img(print_cell_hand, self.print_word, self.hand_word)
        if print_cell_residue:
            merge_print_cell = row_get_pair(merge,print_cell_residue,10)
        merge_residue,print_cell_residue,merge = self.create_big_img(merge_print_cell,merge,print_cell_residue)
        self.row_pairs = merge_residue+merge
        self.hand_after_row_connect = hand_residue
        self.print_after_row_connect = print_cell_residue
        self.all_after_row_connect = self.row_pairs
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
        # self.all_after_row_connect = self.row_pairs
        return self.row_pairs


    def column_connect(self):
        if self.row_pairs:
            self.all_after_row_connect = self.row_pairs+self.hand_after_row_connect+self.print_after_row_connect
        self.column_pairs,_ = column_get_pair(self.all_after_row_connect)
        return self.column_pairs


    def graph_to_forest(self):


        forest_num_list = []
        forest_cell_list = []


        for pair in self.column_pairs:
            top = pair
            bottom = self.column_pairs[pair]
            if len(forest_num_list) == 0:               #当森林里面没有东西时先初始化

                if bottom == -1:                    #-1为没找到配对
                    forest_num_list.append([top])
                else:
                    forest_num_list.append([top,bottom])
            else:
                top_forest = []
                bottom_forest = []

                for i,forest in enumerate(forest_num_list):             #找pair是不是和森林中的树有链接
                    if top in forest:
                        top_forest = forest
                    if bottom in forest:
                        bottom_forest = forest
                    if top_forest and bottom_forest:
                        break

                if top_forest and bottom_forest and top_forest is not bottom_forest:    #判断当前的pair是不是链接两个树
                    top_forest.extend(bottom_forest)                        #将两个树合并,删除一个
                    forest_num_list.remove(bottom_forest)

                elif top_forest:                #如果不是链接,分别加入到自己的树中
                    if bottom!=-1:
                        top_forest.append(bottom)

                elif bottom_forest:
                    bottom_forest.append(top)


                else:                           #如果都没找到,将其作为一颗树放入到深林中
                    if bottom == -1:
                        forest_num_list.append([top])
                    else:
                        forest_num_list.append([top, bottom])


        for forest in forest_num_list:              #讲每个树中所有节点放入一个list中,所有树再放入一个list'中
            cell_forest = []
            for num in forest:
                cell_forest.append(self.all_after_row_connect[num])
            forest_cell_list.append(cell_forest)

        def forest_sort(node):
            return node.top


        for i,forest in enumerate(forest_cell_list):            #讲森林中每个节点排序
            forest.sort(key=forest_sort)
            for j,node in enumerate(forest):
                node.position = (i,j)


        self.forest_list = forest_cell_list



    def judge_fraction(self):
        for forest in self.forest_list:
            for top in forest:
                if top.type == 'merge' or top.type == 'print':
                    num = 0                         #两个打印之间有几个手写框
                    for i in range(top.position[1]+1,len(forest)):
                        bottom = forest[i]
                        if bottom.type == 'merge' or bottom.type == 'print':
                            break
                        elif column_iou(top.bbox,bottom.bbox,'max') >0.5:
                            top.type = 'fraction'








def draw_bbox(bbox,img,x_pro,y_pro,color):
    cv2.rectangle(img, (int(bbox[0] * x_pro), int(bbox[1] * y_pro)),
                  (int(bbox[2] * x_pro), int(bbox[3] * y_pro)), color, 4)






def create_sess():
    g1_config = tf.ConfigProto(allow_soft_placement=True)

    g1 = tf.Graph()
    sess1 = tf.Session(config=g1_config, graph=g1)

    with sess1.as_default():
        with g1.as_default():
            net, run_list = run.build_ctpn_model()
            saver = tf.train.Saver()
            # saver.restore(sess1,
            #               'Arithmetic_Func_detection_for_CTPN_v2_5/checkpoints/VGGnet_fast_rcnn_iter_48000.ckpt')

            saver.restore(sess1,
                          'Arithmetic_Func_detection_for_CTPN_v3/checkpoints/VGGnet_fast_rcnn_iter_90000.ckpt')



    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2)
    with sess2.as_default():
        with g2.as_default():
            inputs = tf.placeholder(tf.float32, [None, 32, None, 1])
            width = tf.placeholder(tf.int32, [None])
            is_training = tf.placeholder(tf.bool)
            model = CTC_Model()
            logits, sequence_length = model.crnn(inputs, width, is_training)

            logits_ = tf.nn.softmax(logits)
            #
            # classs = tf.cast(logits_ > 0.95, tf.float32)
            #
            # logits = logits*classs

            decoders, probably = tf.nn.ctc_beam_search_decoder(logits,
                                                              sequence_length,
                                                              beam_width=20,
                                                              top_paths=5,
                                                              merge_repeated=False)

            decodes_greedy, _ = tf.nn.ctc_greedy_decoder(logits, sequence_length, merge_repeated=True)
            decodes_greedy = decodes_greedy[0]

            dense_greedy_decoder = tf.sparse_to_dense(sparse_indices=decodes_greedy.indices, output_shape=decodes_greedy.dense_shape,
                                               sparse_values=decodes_greedy.values, default_value=-1)

            decoder_list = []

            for decoder in decoders:

                dense_decoder = tf.sparse_to_dense(sparse_indices=decoder.indices, output_shape=decoder.dense_shape,
                                                   sparse_values=decoder.values, default_value=-1)
                decoder_list.append(dense_decoder)

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess2, config.MODEL_SAVE)



            return sess1,sess2,net,run_list,decoder_list,inputs,width,is_training,logits_,sequence_length,dense_greedy_decoder


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






def revise_result(result):

    # state, revise_output, output = delete_top_or_bottom(result)
    state = eval_label(result)   #-------------
    # revise_output = result
    output = result #-----------------------------
    # if state == 'right':
    #     output = revise_output
    #     return 'right',revise_output
    # if state == 'problem':
    #     output, state = add_bracket(output)
    #     if state == 'right':
    #         return 'right',output
    # if state == 'error':
    #     output = pro_problem_to_right(output)
    return state,output

def batch_logits_to_output(logits,sequence_length):
    # logits = logits[..., config.NUM_SIGN]
    # logits = np.swapaxes(logits,0,1)
    output_list = []
    for i in range(logits.shape[0]):
        sequence = sequence_length[i]
        output = logits[i,:sequence,:]
        output = beam_search_decoder(output)
        result_list = []
        for result in output:
            def get_char(num):
                return config.DECODE[num]
            result = map(get_char,result[0])
            # result = re.sub(r'([\d+-×÷=])(\1)+| ', r'\1', ''.join(result))
            result = ''.join(result)
            result_list.append(result)
        output_list.append(result_list)

    return output_list

def logits_to_output(logits,sequence_length):

    sequence = sequence_length
    output = logits[:sequence,:]
    output = beam_search_decoder(output)
    result_list = []
    for result in output:
        def get_char(num):
            return config.DECODE[num]
        result = map(get_char,result[0])
        # result = re.sub(r'([\d+-×÷=])(\1)+| ', r'\1', ''.join(result))
        result = ''.join(result)
        result_list.append(result)

    return result_list


def pipeline(img,sess1,sess2,net, run_list,dense_decoder,inputs,width,is_training,logits,sequence_length,decodes_greedy):

    def sentence_to_output(sentence):
        output = sentence.tolist()

        decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

        output = list(map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))).replace('－', '-'), output))

        return output

    def dection(img,sess1,net, run_list):
        '''
        输出检测结果,一个bbox的list,一个type的list,和检测运行时间
        :param img:
        :param sess1:
        :param net:
        :param run_list:
        :return:
        '''
        tb = time.time()
        feed_dict, img_resized_shape, im_scales, bbox_scale = run.run_ctpn(img, net)
        out_put = sess1.run(run_list, feed_dict)
        bboxes, types = run.decode_ctpn_output(out_put, im_scales, bbox_scale, img_resized_shape)
        te = time.time()
        detect_time = te - tb

        types2 = []
        for i, type in enumerate(types):
            if type == 1:
                types2.append('hand')
            else:
                types2.append('print')

        return bboxes,types2,detect_time



    bboxes, types, detect_time = dection(img,sess1,net, run_list)


    if not types:
        return [],[],[]


    if 'print' in types and 'hand' in types:

        tb = time.time()
        all_result = All_Result(bboxes,img,types)
        all_result.row_connect()
        all_result.column_connect()
        all_result.graph_to_forest()
        all_result.judge_fraction()
        image,wides = all_result.create_input()
        te = time.time()
        layout_time = te-tb

    else:
        tb = time.time()
        all_result = All_Result(bboxes, img, types)
        all_result.column_connect()
        all_result.graph_to_forest()
        # all_result.judge_fraction()
        image, wides = all_result.create_input()
        te = time.time()
        layout_time = te - tb



    tb = time.time()
    if config.GREEADY_BEAM:
        logits, decodes_greedy,sequence_length = sess2.run([logits, decodes_greedy,sequence_length],
                                            feed_dict={inputs: image, width: wides, is_training: False})
        output_list = sentence_to_output(decodes_greedy)
        logits = logits[..., config.NUM_SIGN]
        logits = np.swapaxes(logits,0,1)
    else:
        sentence_list = sess2.run(dense_decoder, feed_dict={inputs: image, width: wides, is_training: False})
        output_list = [sentence_to_output(sentence) for sentence in sentence_list]

    te = time.time()

    recognition_time = te-tb





    tb = time.time()
    for i, result in enumerate(all_result.all_after_row_connect):

        if config.GREEADY_BEAM:
            result.output = output_list[i]
        else:
            #----------------------------------------------------------------beam_search
            result.output = output_list[0][i]
            result.backup_output = [output_[i] for output_ in output_list]


        '''
        等式的判题
        '''
        if result.type == 'merge':
            result.state,result.output = revise_result(result.output)
            if result.state != 'right' and no_chinese(result.output):
                if config.GREEADY_BEAM:
                    result.backup_output = logits_to_output(logits[i],sequence_length[i])
                for output in result.backup_output:
                    state,output_temp = revise_result(output)
                    if state == 'right':
                        result.state = state
                        result.output = output_temp
                        break
            all_result.connect_result.append(result)




        elif result.type == 'print':
            state,output = revise_result(result.output)
            if state == 'right':
                result.state = state
                result.output = output
                all_result.connect_result.append(result)
            elif '@' in result.output:
                all_result.fraction.append(result)
            else:
                all_result.vertical.append(result)


        elif result.type == 'hand':
            state, output = revise_result(result.output)
            if state == 'right':
                result.state = state
                result.output = output
                all_result.connect_result.append(result)

            else:
                all_result.other_result.append(result)


        else:
            all_result.other_result.append(result)




    for result in all_result.vertical:
        # if result.output == '350-8×6+5×4':
        #     print('a')
        bottom_list = []
        position = result.position
        forest = all_result.forest_list[position[0]]
        for num in range(max(position[1]-2,0), len(forest)):
            bottom = forest[num]
            if bottom.type == 'print' or bottom.type == 'merge':
                #break
                continue
            else:
                label = result.output.replace('=', '') + '=' + bottom.output.replace('=', '')
                state, output = revise_result(label)
                for result_output in result.backup_output:
                    break_out = False
                    for bottom_output in bottom.backup_output:
                        label = result_output.replace('=', '') + '=' + bottom_output.replace('=', '')
                        state, output = revise_result(label)
                        if state == 'right':
                            break_out = True
                            break
                    if break_out:
                        break

                bottom_list.append(bottom)
                if state == 'right':
                    result = Result([min(result.left,bottom.left),min(result.top,bottom.top),max(result.right,bottom.right),max(result.bottom,bottom.bottom)],
                                    all_result.img,type='merge')
                    result.output = output
                    result.state = state
                    all_result.connect_result.append(result)
                    break
        if result.state != 'right':
            bottom_list.append(result)
            if no_chinese(result.output) and set('+-×÷=')&set(result.output) :
                result.state = 'error'
            else:
                result.state = 'problem'
            all_result.connect_result.append(result)
    te = time.time()
    after_processing_time = te-tb




    # print('检测时间:{}'.format(detect_time))
    # print('版面分析:{}'.format(layout_time))
    # print('识别时间:{}'.format(recognition_time))
    # print('后处理时间{}'.format(after_processing_time))



    #
    # for result in all_result.vertical:
    #
    #     bottom_list = []
    #     position = result.position
    #     forest = all_result.forest_list[position[0]]
    #     for num in range(position[0]+1, len(forest)):
    #         bottom = forest[num]
    #         if bottom.type == 'print' or bottom.type == 'merge':
    #             break
    #
    #         else:
    #             label = result.output.replace('=', '') + '=' + bottom.output.replace('=', '')
    #
    #             state, output = revise_result(label)
    #             bottom_list.append(bottom)
    #             if state == 'right':
    #                 result.output = output
    #                 result.state = state
    #                 all_result.connect_result.append(result)
    #                 break
    #
    #     if result.state != 'right':
    #         bottom_list.append(result)
    #         result.left = min([bottom.left for bottom in bottom_list])
    #         result.right =  max([bottom.right for bottom in bottom_list])
    #         result.top = min([bottom.top for bottom in bottom_list])
    #         result.bottom = max([bottom.bottom for bottom in bottom_list])
    #         result.bbox = [result.left,result.top,result.right,result.bottom]
    #         result.state = 'error'
    #         all_result.connect_result.append(result)

    return all_result,bboxes,types


def area(point1,point2):
    return max((point2[0]-point1[0]),0)*max((point2[1]-point1[1]),0)





def get_iou(box1,box2):
    area1 = area((box1[0],box1[1]),(box1[2],box1[3]))
    area2 = area((box2[0],box2[1]),(box2[2],box2[3]))

    point1 = (max(box1[0],box2[0]),max(box1[1],box2[1]))
    point2 = (min(box1[2],box2[2]),min(box1[3],box2[3]))

    area3 = area(point1,point2)

    return area3/(area1+area2-area3)

def pair_check(img,sess1,sess2,net, run_list,dense_decoder,inputs,width,is_training):

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
    all_result.row_connect_test()
    cv2.imwrite('xxx.jpg',all_result.img)
    cv2.imshow('a',all_result.img)
    cv2.waitKey()



def no_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    return True


def result_test(all_img_path,save_path,model='pipline'):
    if model == 'pipline':
        sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training,logits,sequence_length,decodes_greedy = create_sess()
        for img_path in tqdm(glob(os.path.join(all_img_path,'*'))):
            img = cv2.imread(img_path)
            img_save_path = os.path.join(save_path,img_path.split('/')[-1].split('.')[0])
            if not os.path.exists(img_save_path):
                os.mkdir(img_save_path)
            all_result,bboxes,types = pipeline(img.copy(), sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training,logits,sequence_length,decodes_greedy)
            for i,result in enumerate(all_result.connect_result):
                if result.state != 'right' and no_chinese(result.output):
                    cv2.imwrite(os.path.join(img_save_path,str(i)+'_'+result.output+'.jpg'),result.img)

            x_pro = 3024 / img.shape[1]
            y_pro = 4031 / img.shape[0]
            img = cv2.resize(img, (3024, 4032))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2 = img.copy()
            draw_bboxes(img, all_result.connect_result+all_result.other_result, x_pro, y_pro)
            img = draw_result(img, all_result.connect_result+all_result.other_result, x_pro, y_pro)

            for i,box in enumerate(bboxes):
                if types[i] == 'print':
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                draw_bbox(box, img2, x_pro, y_pro, color)
            big_img_path = os.path.join(save_path, img_path.split('/')[-1])
            img.save(big_img_path.replace('.', '_.'))
            cv2.imwrite(big_img_path.replace('.', '_2.'), img2)
    else:
        pass



def run_single_img():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='image path', type=str)
    args = parser.parse_args()
    img = cv2.imread(args.path)
    # #
    #
    sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training, logits, sequence_length, decodes_greedy = create_sess()
    # pair_check(img.copy(), sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training)
    time1 = time.time()
    result, bboxes, types = pipeline(img.copy(), sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training,
                                     logits, sequence_length, decodes_greedy)
    time2 = time.time()
    print('总共耗时{}'.format(time2 - time1))
    x_pro = 3024 / img.shape[1]
    y_pro = 4031 / img.shape[0]
    img = cv2.resize(img, (3024, 4032))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = img.copy()
    draw_bboxes(img, result.connect_result+result.other_result, x_pro, y_pro)

    for i, box in enumerate(bboxes):
        if types[i] == 'print':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        draw_bbox(box, img2, x_pro, y_pro, color)

    img = draw_result(img, result.connect_result+result.other_result, x_pro, y_pro)
    img.show()
    img.save(args.path.replace('.', '_.'))
    cv2.imwrite(args.path.replace('.', '_2.'), img2)


if __name__ == '__main__':
    run_single_img()

    # result_test('/home/wzh/竖式，脱式/img','/home/wzh/pipline_result/shushi4')
    #




    #








    # sess1, sess2, net, run_list, dense_decoder,inputs,width,is_training ,logits,sequence_length,decodes_greedy= create_sess()
    # img_path = glob('/home/wzh/test16/*')
    # for path in tqdm(img_path):
    #     img = cv2.imread(path)
    #     save_path = path.replace('test16','pipline_result/test16_3')
    #
    #     sess1, sess2, net, run_list, dense_decoder,inputs,width,is_training ,logits,sequence_length,decodes_greedy= create_sess()
    #     time1 = time.time()
    #     result,bboxes,types = pipeline(img.copy(), sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training,logits,sequence_length,decodes_greedy)
    #     time2 = time.time()
    #     print('总共耗时{}'.format(time2-time1))
    #     print('------------------------------------------------------------------------------------------------------------------------------------')
        # x_pro = 3024 / img.shape[1]
        # y_pro = 4031 / img.shape[0]
        # img = cv2.resize(img, (3024, 4032))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # # draw_box(img,result.right_error_result,x_pro,y_pro)
        # # img = draw_result(img,result.right_error_result,x_pro,y_pro)
        # # draw_box(img,result.problem_result,x_pro,y_pro)
        # # img = draw_result(img,result.problem_result,x_pro,y_pro)
        #
        # draw_box(img, result.connect_result, x_pro, y_pro)
        # img = draw_result(img, result.connect_result, x_pro, y_pro)
        #
        # # draw_box(img, result.results, x_pro, y_pro)
        # # img = draw_result(img, result.results, x_pro, y_pro)
        #
        # img.save(save_path)
