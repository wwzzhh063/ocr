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
# from easytest import beam_search_decoder
from layout_utils import row_get_pair,no_chinese,column_get_pair
from utils import image_size_normal,eval_label,draw_bboxes,draw_result,draw_pair,draw_column_pair









class Bbox(object):
    def __init__(self,bbox,label,type):
        self.bbox = bbox
        self.label = label
        self.output = label
        self.type = type
        self.state = 'start'
        self.classes = '='


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

        self.centre = (int((self.left + self.right) / 2), int((self.top + self.bottom) / 2))





class Img_ALL_BBox(object):
    def __init__(self,img,name):
        self.print_word = []
        self.hand_word = []
        self.merge = []
        self.img_path = ''
        self.pair = {}
        self.img = img
        self.not_pair = []
        self.all_box = []
        self.problem_box = []
        self.problem_label = []
        self.error_label = []
        self.right_label = []
        self.name = name

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

            label = print_cell.label + hand.label
            big_img = Bbox([left, top, right, bottom],label,'merge')
            if '*' in big_img.label or '~' in big_img.label:
                big_img.classes = '...'


            merge.append(big_img)

        return box_list1_, box_list2_, merge



    def create_big_img2(self,pair,box_list1,box_list2):         #合并括号填词的
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

            label = print_cell.label + hand.label
            big_img = Bbox([left, top, right, bottom], label, 'merge')
            big_img.classes = '()'

            merge.append(big_img)

        return box_list1_, box_list2_, merge



    def row_connect(self):
        print_cell_hand = row_get_pair(self.print_word, self.hand_word)
        self.print_hand_pair = print_cell_hand
        print_cell_residue, hand_residue, merge = self.create_big_img(print_cell_hand, self.print_word, self.hand_word)
        if print_cell_residue:
            merge_print_cell = row_get_pair(merge, print_cell_residue)
        else:
            merge_print_cell = {}
        self.bracket_pair = merge_print_cell
        self.merge = merge
        self.print_not_pair = print_cell_residue
        merge_residue, print_cell_residue, merge = self.create_big_img2(merge_print_cell, merge, print_cell_residue)
        self.row_pairs = merge_residue + merge
        self.hand_after_row_connect = hand_residue
        self.print_after_row_connect = print_cell_residue
        self.all_after_row_connect = self.row_pairs


        self.check_label()


    def revise_label(self,bbox):
        '''
        修改标注一些可能出现的问题
        :return:
        '''
        label = bbox.label
        if label.count('=') >1:
            label ='='.join([label.split('=')[0],label.split('=')[-1]])
            bbox.label = label
            bbox.output = label
            bbox.state = eval_label(label)

    def check_label(self):
        '''
        检查匹配的算式中是否含有problem的算式
        检查未匹配的算式中是否含有手写且不是中文的框
        :return:
        '''

        for bbox in self.row_pairs:
            state = eval_label(bbox.label)

            if state == 'problem':
                self.revise_label(bbox)
                if bbox.state == 'problem':
                    self.problem_label.append(bbox)
            elif state == 'error':
                bbox.state = state
                self.error_label.append(bbox)


            else:
                bbox.state = state
                self.right_label.append(bbox)

        for bbox in self.print_after_row_connect:
            if no_chinese(bbox.label) and bbox.type == 'print':
                self.problem_label.append(bbox)

        self.all_box = self.error_label+self.right_label



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





def find_labels(name,j,recog_path):
    name = name.split('.')[0]
    name = name.split('_')[-1]
    path = recog_path
    paths = glob(os.path.join(path,'*'))
    label_path = ''
    for i in paths:
        # if name == '201809041126441' :
        #     print('a')
        if name in i:
            '''
            名字在路径里面不一定是对应的标签
            比如'微信图片_201809041550286朱会敏'和'微信图片_20180904155028朱会敏',只是他的前缀
            '''
            temp = i.split(name)[-1]
            if not temp:
                #temp为空
                label_path = i
                break

            elif  temp[0] not in ['0','1','2','3','4','5','6','7','8','9']:
                label_path = i
                j = j+1
                break


    return j,label_path


def set_xml_data(dection_xml,dection_img,recog_path,recognition_xml='outputs'):
    '''

    :param dection_xml: 检测的xml文件夹地址
    :param dection_img: 检测的图片地址
    :param recognition_xml: 识别的xml文件夹名称
    :return:
    '''


    all_dection_xml_path = glob(os.path.join(dection_xml, '*.xml'))
    dection_img_name = []                   #所有图片名称
    dection_all_img_label  = []                     #所有图片的标签

    j = 0
    for all_dection_xml_path in tqdm(all_dection_xml_path):            #所有xml
        p = ParseXml(all_dection_xml_path)
        img_name_, class_list, bbox_list,jpg_or_JPG = p.get_bbox_class()
        big_img_path = os.path.join(dection_img,all_dection_xml_path.split('/')[-1].replace('xml',jpg_or_JPG))
        all_bbox_img = Img_ALL_BBox(cv2.imread(big_img_path),img_name_)           #一张图片中的所有box   tetst---------------------------------
        all_bbox_img.img_path = big_img_path
        dection_img_name.append(img_name_)
        j,path = find_labels(img_name_,j,recog_path)                 #找到对应的切分图片的地址



        if path != '':
            small_img_xml = sorted(glob(os.path.join(path,recognition_xml+'/*')))            #得到一张图片所有的xml
            for i,type in enumerate(class_list):                 #一张图片中每一个box

                small_img_xml_path = '/'.join(small_img_xml[0].split('/')[0:-1])+'/'+str(i).zfill(5)+'.xml'

                if type == 1:
                    type = 'print'
                else:
                    type = 'hand'

                if os.path.exists(small_img_xml_path):
                     label,name = from_xml_read_label.read_label(small_img_xml_path)
                     bbox = Bbox(bbox_list[i],label,type)
                     # bbox = Bbox_test(bbox_list[i], label, type,small_img_xml_path)          #for test!!!!!!!!!!
                     if type == 'print':
                         all_bbox_img.print_word.append(bbox)
                     else:
                         all_bbox_img.hand_word.append(bbox)

            dection_all_img_label.append(all_bbox_img)

    return dection_all_img_label


def output_check_result(save_path,xml_path,img_path,recog_path,recognition_xml):
    '''
    将含有问题算式的图片输出出来
    1.标注错误
    2.版面分析错误
    :param save_path: 保存输出结果的地址
    :return:
    '''


    all_img = set_xml_data(xml_path, img_path,recog_path,recognition_xml)


    for i, img_result in tqdm(enumerate(all_img)):
        img_result.row_connect()


        column_pairs = img_result.column_connect()
        img_result.graph_to_forest()


        if img_result.problem_label :

            save_path1 = os.path.join(save_path,'problem')

            if not os.path.exists(save_path1):
                os.mkdir(save_path1)

            img = img_result.img
            img,x_pro, y_pro = image_size_normal(img)

            img2 = img.copy()

            draw_bboxes(img,img_result.problem_label, x_pro,y_pro)
            img = draw_result(img,img_result.problem_label,x_pro,y_pro)
            img.save(os.path.join(save_path1,img_result.img_path.split('/')[-1]))

            draw_bboxes(img2, img_result.print_word+img_result.hand_word, x_pro, y_pro)
            draw_pair(img_result.print_hand_pair,img_result.print_word,img_result.hand_word,img2,(255,0,0),x_pro, y_pro)


            draw_pair(img_result.bracket_pair,img_result.merge,img_result.print_not_pair,img2,(0,0,255),x_pro, y_pro)


            # draw_column_pair(column_pairs,img_result.all_after_row_connect,img2,x_pro, y_pro)
            img2 = draw_result(img2, [], x_pro, y_pro)
            img2.save(os.path.join(save_path1, img_result.img_path.split('/')[-1].replace('.','_.')))



        if (img_result.problem_label+ img_result.error_label) :


            save_path2 = os.path.join(save_path, 'problem_error')
            if not os.path.exists(save_path2):
                os.mkdir(save_path2)

            img = img_result.img
            img,x_pro, y_pro = image_size_normal(img)

            img2 = img.copy()

            draw_bboxes(img,img_result.problem_label+img_result.error_label, x_pro,y_pro)
            img = draw_result(img,img_result.problem_label+img_result.error_label,x_pro,y_pro)
            img.save(os.path.join(save_path2,img_result.img_path.split('/')[-1]))

            draw_bboxes(img2, img_result.print_word+img_result.hand_word, x_pro, y_pro)
            draw_pair(img_result.print_hand_pair,img_result.print_word,img_result.hand_word,img2,(255,0,0),x_pro, y_pro)


            draw_pair(img_result.bracket_pair,img_result.merge,img_result.print_not_pair,img2,(0,0,255),x_pro, y_pro)


            # draw_column_pair(column_pairs,img_result.all_after_row_connect,img2,x_pro, y_pro)
            img2 = draw_result(img2, [], x_pro, y_pro)
            img2.save(os.path.join(save_path2, img_result.img_path.split('/')[-1].replace('.','_.')))




       





def to_int(a):
    x,y = a
    return (int(x),int(y))


def get_box_centre(box):
    if len(box) == 4:
        centre = ((box[0]+box[2])/2,(box[1]+box[3])/2)
    else:
        centre = ((box[0]+box[4])/2,(box[1]+box[5])/2)
    return to_int(centre)


def get_centre(box):
    if len(box) == 4:
        point = (int((box[2]+box[0])/2), int((box[1] + box[3]) / 2))
    else:
        point = (int((box[0] + box[4]) / 2), int((box[1] + box[5]) / 2))

    return point




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



def create_dataset(xml_path):


    all_img = set_xml_data(xml_path)


    for i,img_result in tqdm(enumerate(all_img)):
        img = cv2.imread(img_result.img_path)
        resize_list = []

        if img.shape[0]>3900:
            resize_list.append((1920/img.shape[0],1080/img.shape[1]))
            resize_list.append((854/img.shape[0],640/img.shape[0]))
        elif img.shape[0]>1800:
            resize_list.append((854/img.shape[0],640/img.shape[0]))

        img_result.create_pair()
        #
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



if __name__ == '__main__':

    save_path = os.environ['HOME']+'/第五批-测试集/第五批测试集-检验'               #保存的地址
    xml_path = os.environ['HOME']+'/第五批-测试集/第五批测试集/生成的xml文件'          #验证集检测标注的文件夹
    img_path = os.environ['HOME']+'/第五批-测试集/第五批测试集/原始图片'              #验证集的图片文件夹
    recog_path = os.environ['HOME']+'/第五批-测试集/第五批测试集识别图-result'           #验证集的识别标注文件夹
    recognition_xml = 'xml'                                             #验证集识别xml文件夹的名称



    output_check_result(save_path,xml_path,img_path,recog_path,recognition_xml)

    # output_check_result('/home/wzh/第一批/val的验证','/home/wzh/第一批/img_val_xml','/home/wzh/第一批/img_val','/home/wzh/第一批/suanshi_val','outputs')
    # output_check_result('output_check_result2','/home/wzh/第一批/img_train_xml','/home/wzh/第一批/img_train','/home/wzh/第一批/suanshi_train','outputs')





