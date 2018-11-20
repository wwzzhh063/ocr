from pipline import pipeline,create_sess
from glob import glob
from tqdm import tqdm
import os
import re
from inference import set_xml_data
import Levenshtein
from layout_utils import row_get_pair,column_get_pair,column_iou
from utils import draw_pair,eval_label,draw_bboxes,draw_result,get_iou
from math import log
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont


class Single_Img_Evaluate(object):
    def __init__(self,true_result,pre_result,img,name,save_path,log_path,bboxes, types):
        self.equation_all = 0
        self.equation_right = 0
        self.bracket_all = 0
        self.bracket_right = 0
        self.residual_all = 0
        self.residual_right = 0
        self.state_all = 0
        self.state_right = 0
        self.char_acc_all = 0
        self.char_acc = 0
        self.recall = 0
        self.all_num = 0
        self.error = []
        self.not_recall = []
        self.true_result = true_result
        self.pre_result = pre_result
        self.img = img
        self.name = name
        self.save_path = save_path
        self.log_path = log_path
        self.bboxes = bboxes
        self.types = types


    def compute(self):
        self.all = self.residual_all+self.bracket_all+self.equation_all
        self.right = self.residual_right+self.bracket_right+self.equation_right
        self.seq_acc = self.right/self.all
        self.state_acc = self.state_right/self.state_all
        self.char_acc = self.char_acc_all / self.all
        self.all_recall = self.recall / self.all_num


        if self.equation_all == 0:
            self.equation_acc = 0
        else:
            self.equation_acc = self.equation_right/self.equation_all
        if self.bracket_all == 0:
            self.bracket_acc = 0
        else:
            self.bracket_acc = self.bracket_right/self.bracket_all
        if self.residual_all == 0:
            self.residual_acc = 0
        else:
            self.residual_acc = self.residual_right/self.residual_all


        self.evaluate_dict = {'all':self.seq_acc,'=':self.equation_acc,'()':self.bracket_acc,'...':self.residual_acc,'state':self.state_acc,'char_acc':
                              self.char_acc,'recall':self.all_recall}


    def print_write_result(self,log_path):
        print('图片名称:{}'.format(self.name))
        print('总共:{}道题,正确率:{}'.format(self.all,self.seq_acc))
        print('等式:{}道题,正确率:{}'.format(self.equation_all,self.equation_acc))
        print('填空题:{}道题,正确率:{}'.format(self.bracket_all,self.bracket_acc))
        print('求余数:{}道题,正确率:{}'.format(self.residual_all,self.residual_acc))
        print('判断对错的正确率{}'.format(self.state_acc))
        print('字符正确率{}'.format(self.char_acc))
        print('召回率{}'.format(self.all_recall))

        log = open(log_path, 'a')
        log.writelines('图片名称:{}\n'.format(self.name))
        log.writelines('总共:{}道题,正确率:{}\n'.format(self.all,self.seq_acc))
        log.writelines('等式:{}道题,正确率:{}\n'.format(self.equation_all,self.equation_acc))
        log.writelines('填空题:{}道题,正确率:{}\n'.format(self.bracket_all,self.bracket_acc))
        log.writelines('求余数:{}道题,正确率:{}\n'.format(self.residual_all,self.residual_acc))
        log.writelines('判断对错的正确率{}\n'.format(self.state_acc))
        log.writelines('字符正确率{}\n'.format(self.char_acc))
        log.writelines('召回率{}\n'.format(self.all_recall))
        log.writelines('-------------------------------------------------------------------------------------------------------------\n')



    def get_pair(self):
        self.result_pair = {}

        for i,true_box in enumerate(self.true_result.all_box):
            max_iou = 0
            pair = -1
            for j,pre_box in enumerate(self.pre_result.connect_result):
                iou = get_iou(pre_box.bbox,true_box.bbox)
                if iou>max_iou:
                    max_iou = iou
                    pair = j

            if max_iou>0.5:
                self.result_pair[i] = pair

            else:
                self.result_pair[i] = -1


        return self.result_pair


    def statistic_data(self):

        for true_num in self.result_pair:

            true_box = self.true_result.all_box[true_num]

            if self.result_pair[true_num] != -1:
                pre_box = self.pre_result.connect_result[self.result_pair[true_num]]

                if true_box.classes == '=':
                    if true_box.label == pre_box.output:
                        self.equation_right = self.equation_right+1
                    else:
                        self.error.append([true_box,pre_box])
                    self.equation_all = self.equation_all+1

                elif true_box.classes == '()':
                    if true_box.label == pre_box.output:
                        self.bracket_right = self.bracket_right+1
                    else:
                        self.error.append([true_box, pre_box])
                    self.bracket_all = self.bracket_all+1

                else:
                    if true_box.label == pre_box.output:
                        self.residual_right = self.residual_right+1
                    elif true_box.label.replace('*','') == pre_box.output.replace('*',''):
                        self.residual_right = self.residual_right + 1
                    else:
                        self.error.append([true_box, pre_box])
                    self.residual_all = self.residual_all+1

                self.char_acc_all = self.char_acc_all + 1 - (Levenshtein.distance(true_box.label, pre_box.output) / len(true_box.label))

                if true_box.state == pre_box.state:
                    self.state_right = self.state_right+1
                self.state_all = self.state_all+1

                self.recall = self.recall+1

            else:
                self.not_recall.append(true_box)

            self.all_num = self.all_num+1


    def draw_result(self):
        x_pro = 3024 / self.img.shape[1]
        y_pro = 4031 / self.img.shape[0]
        img = cv2.resize(self.img, (3024, 4032))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ttfont = ImageFont.truetype('SimSun.ttf', 50)
        img2 = img.copy()

        if self.error:
            for error_result in self.error:
                true = error_result[0]
                pre = error_result[1]
                cv2.rectangle(img, (int(true.left * x_pro), int(true.top * y_pro)),
                              (int(true.right * x_pro), int(true.bottom * y_pro)), (0,255,0), 4)
                cv2.rectangle(img, (int(pre.left * x_pro), int(pre.top * y_pro)),
                              (int(pre.right * x_pro), int(pre.bottom * y_pro)), (255,0,0), 4)

                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                draw.text((int(true.left * x_pro), int(true.top * y_pro - 50)), true.output, fill='green',
                          font=ttfont)
                draw.text((int(pre.left * x_pro), int(pre.bottom * y_pro - 50)), pre.output, fill='red',
                          font=ttfont)

                img = np.asarray(img)


        if self.not_recall:
            for result in self.not_recall:
                cv2.rectangle(img, (int(result.left * x_pro), int(result.top * y_pro)),
                              (int(result.right * x_pro), int(result.bottom * y_pro)), (0, 0, 255), 4)
                cv2.rectangle(img2, (int(result.left * x_pro), int(result.top * y_pro)),
                              (int(result.right * x_pro), int(result.bottom * y_pro)), (0, 0, 255), 4)
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                draw.text((int(result.left * x_pro), int(result.top * y_pro - 50)), result.output, fill='blue',
                          font=ttfont)

                img = np.asarray(img)

            for i, bbox in enumerate(self.bboxes):
                if self.types[i] == 'print':
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(img2, (int(bbox[0] * x_pro), int(bbox[1] * y_pro)),
                              (int(bbox[2] * x_pro), int(bbox[3] * y_pro)), color, 4)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        if self.error:
            cv2.imwrite(os.path.join(self.save_path,self.name+'.jpg'),img)
        if self.not_recall:
            cv2.imwrite(os.path.join(self.save_path, self.name + '_.jpg'), img2)



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
        self.char_acc_all = 0
        self.recall = 0
        self.all_num = 0

    def add_data(self,single_img_evaluate):
        self.equation_all = self.equation_all + single_img_evaluate.equation_all
        self.equation_right = self.equation_right + single_img_evaluate.equation_right
        self.bracket_all = self.bracket_all + single_img_evaluate.bracket_all
        self.bracket_right = self.bracket_right + single_img_evaluate.bracket_right
        self.residual_all = self.residual_all + single_img_evaluate.residual_all
        self.residual_right = self.residual_right + single_img_evaluate.residual_right
        self.state_all =  self.state_all + single_img_evaluate.state_all
        self.state_right =  self.state_right + single_img_evaluate.state_right
        self.char_acc_all = self.char_acc_all + single_img_evaluate.char_acc_all
        self.recall = self.recall + single_img_evaluate.recall
        self.all_num = self.all_num + single_img_evaluate.all_num

    def compute(self):
        self.all = self.residual_all+self.bracket_all+self.equation_all
        self.right = self.residual_right+self.bracket_right+self.equation_right
        self.seq_acc = self.right/self.all
        self.state_acc = self.state_right/self.state_all
        self.char_acc = self.char_acc_all / self.all
        self.all_recall = self.recall / self.all_num


        if self.equation_all == 0:
            self.equation_acc = 0
        else:
            self.equation_acc = self.equation_right/self.equation_all
        if self.bracket_all == 0:
            self.bracket_acc = 0
        else:
            self.bracket_acc = self.bracket_right/self.bracket_all
        if self.residual_all == 0:
            self.residual_acc = 0
        else:
            self.residual_acc = self.residual_right/self.residual_all


        self.evaluate_dict = {'all':self.seq_acc,'=':self.equation_acc,'()':self.bracket_acc,'...':self.residual_acc,'state':self.state_acc,'char_acc':
                              self.char_acc,'recall':self.all_recall}


    def print_result(self):
        print('总共:{}道题,正确率:{}'.format(self.all,self.seq_acc))
        print('等式:{}道题,正确率:{}'.format(self.equation_all,self.equation_acc))
        print('填空题:{}道题,正确率:{}'.format(self.bracket_all,self.bracket_acc))
        print('求余数:{}道题,正确率:{}'.format(self.residual_all,self.residual_acc))
        print('判断对错的正确率{}'.format(self.state_acc))
        print('字符正确率{}'.format(self.char_acc))
        print('召回率{}'.format(self.all_recall))

    def write_result(self,log_path):
        log = open(log_path, 'a')
        log.writelines('总共:{}道题,正确率:{}\n'.format(self.all, self.seq_acc))
        log.writelines('等式:{}道题,正确率:{}\n'.format(self.equation_all, self.equation_acc))
        log.writelines('填空题:{}道题,正确率:{}\n'.format(self.bracket_all, self.bracket_acc))
        log.writelines('求余数:{}道题,正确率:{}\n'.format(self.residual_all, self.residual_acc))
        log.writelines('判断对错的正确率{}\n'.format(self.state_acc))
        log.writelines('字符正确率{}\n'.format(self.char_acc))
        log.writelines('召回率{}\n'.format(self.all_recall))
        log.writelines('-------------------------------------------------------------------------------------------------------------\n')





def evaluate(save_path,log_path,xml_path, img_path,recog_path,recognition_xml):


    all_img = set_xml_data(xml_path, img_path,recog_path,recognition_xml)

    sess1, sess2, net, run_list, dense_decoder, inputs, width, is_training, logits, sequence_length, decodes_greedy = create_sess()

    evaluate_data = Evaluate_Data()

    for i,true_result in enumerate(all_img):
        true_result.row_connect()
        img = true_result.img
        pre_result, bboxes, types = pipeline(img.copy(), sess1, sess2, net, run_list, dense_decoder, inputs, width,
                                         is_training, logits, sequence_length, decodes_greedy)
        single_img_evaluate = Single_Img_Evaluate(true_result,pre_result,img,true_result.name,save_path,'',bboxes, types)
        single_img_evaluate.get_pair()
        single_img_evaluate.statistic_data()
        single_img_evaluate.compute()
        single_img_evaluate.print_write_result(log_path)
        single_img_evaluate.draw_result()
        print('目前总共:')
        evaluate_data.add_data(single_img_evaluate)
        evaluate_data.compute()
        evaluate_data.print_result()
        print('------------------------------------------------------------------')

    evaluate_data.write_result(log_path)


if __name__ == '__main__':
    save_path = '/home/wzh/第一批/val的验证'          #保存的地址
    log_path = 'log.txt'                            #结果的log文件夹
    xml_path = '/home/wzh/第一批/img_val_xml'          #验证集检测标注的文件夹
    img_path = '/home/wzh/第一批/img_val'              #验证集的图片文件夹
    recog_path = '/home/wzh/第一批/suanshi_val'        #验证集的识别标注文件夹
    recognition_xml = 'outputs'                         #验证集识别xml文件夹的名称




    #第五批------------------------------------------------------------------------------------------------------
    save_path = '/home/wzh/第五批-测试集/第五批测试集-检验'
    log_path = '/home/wzh/第五批-测试集/第五批测试集-检验/log.txt'
    xml_path = '/home/wzh/第五批-测试集/第五批测试集/生成的xml文件'
    img_path = '/home/wzh/第五批-测试集/第五批测试集/原始图片'
    recog_path = '/home/wzh/第五批-测试集/第五批测试集识别图-result'
    recognition_xml = 'xml'


    evaluate(save_path,log_path,xml_path, img_path,recog_path,recognition_xml)