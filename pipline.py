import utils
import tensorflow as tf
from model import CTC_Model
from config import Config as config
import cv2
from Arithmetic_Func_detection_for_CTPN_v1.ctpn import run
from PIL import Image,ImageDraw,ImageFont
import math
import argparse
import time
from glob import glob
from tqdm import tqdm
import os

class Result(object):
    def __init__(self,bbox,img):
        self.top = bbox[1]
        self.bottom = bbox[3]
        self.left = bbox[0]
        self.right = bbox[2]
        self.img = img[self.top:self.bottom+1,self.left:self.right+1,...]
        self.normal_img = utils.image_normal(self.img.copy())
        self.img_wide = self.normal_img.shape[1]
        self.output = ''
        self.revise_output = ''
        self.state = 'right'

    def set_result(self,result):
        self.result = result


class All_Result(object):
    def __init__(self,bboxes,img):
        self.bboxes = bboxes
        self.img = img
        self.results = []
        self.connect_result = []
        self.max_wide = 0
        self.ave_wide = 0
        self.ave_high = 0
        self.right_error_result = []
        self.problem_result = []

    def create_input(self):
        imgs = []
        wides = []

        for box in self.bboxes:
            result = Result(box,self.img)
            if result.img_wide>self.max_wide:
                self.max_wide = result.img_wide
            wides.append(result.img_wide)
            imgs.append(result.normal_img)
            self.results.append(result)

        inputs,wides = utils.create_input(imgs,self.max_wide,wides)

        return inputs,wides

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
            rgb = (255,0,0)
            cv2.rectangle(img,(int(result.left*x_pro),int(result.top*y_pro)),(int(result.right*x_pro),int(result.bottom*y_pro)),rgb,4)

def draw_result(img,all_result,x_pro,y_pro):
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




def pipline2(img,sess1,sess2,net, run_list,dense_decoder,inputs,width,is_training):
    feed_dict, img_resized_shape, im_scales, bbox_scale = run.run_ctpn(img, net)
    out_put = sess1.run(run_list, feed_dict)

    bboxes = run.decode_ctpn_output(out_put, im_scales, bbox_scale, img_resized_shape)


    all_result = All_Result(bboxes,img)
    image,wides = all_result.create_input()

    sentence = sess2.run(dense_decoder, feed_dict={inputs: image, width: wides, is_training: False})

    output = sentence.tolist()

    decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

    output = list(map(lambda y: ''.join(list(map(lambda x: decode.get(x), y))), output))

    problems = []

    ave_wide = 0
    ave_high = 0

    for i, result in enumerate(all_result.results):
        ave_wide = ave_wide + result.right - result.left
        ave_high = ave_high + result.bottom - result.top


        try:
            # result.state = eval_label(output[i])
            result.state, result.revise_output, result.output = delete_top_or_bottom(output[i])
        except:
            result.state = 'problem'

        if result.state == 'problem':
            all_result.problem_result.append(result)
        else:
            all_result.right_error_result.append(result)

    all_result.ave_wide = all_result.ave_wide / len(all_result.results)
    all_result.ave_high = all_result.ave_high / len(all_result.results)

    temp = all_result.problem_result.copy()

    delete = []
    for i, problem1 in enumerate(temp):
        problems_without = temp.copy()
        problems_without.pop(i)

        for j, problem2 in enumerate(problems_without):
            # if problem1.output == '8458-112=' and problem2.output == '3461':
            #     print('aaaaaaa')
            if min(problem1.bottom, problem2.bottom) > max(problem1.top, problem2.top):  # 判断在一行
                # label = problem1.output+problem2.output
                # state,revise_output,output = delete_top_or_bottom(label)
                state, revise_output, output = delete_pair_problem_result(problem1.output, problem2.output)
                if state != 'problem':
                    box = [min(problem1.left, problem2.left), min(problem1.top, problem2.top),
                           max(problem1.right, problem2.right), max(problem1.bottom, problem2.bottom)]
                    result = Result(box, img)
                    result.state = state
                    result.revise_output = revise_output
                    result.output = output
                    all_result.right_error_result.append(result)
                    try:
                        all_result.problem_result.remove(problem1)
                        all_result.problem_result.remove(problem2)
                    except:
                        pass

                    # all_result.problem_result.pop(i)
                    # if i<=j:
                    #     all_result.problem_result.pop(j+1)
                    # else:
                    #     all_result.problem_result.pop(j)
                    # print(output)

    all_result.connect_result.extend(all_result.problem_result)
    all_result.connect_result.extend(all_result.right_error_result)

    return all_result


def create_sess():
    g1_config = tf.ConfigProto(allow_soft_placement=True)

    g1 = tf.Graph()
    sess1 = tf.Session(config=g1_config, graph=g1)

    with sess1.as_default():
        with g1.as_default():
            net, run_list = run.build_ctpn_model()
            saver = tf.train.Saver()
            saver.restore(sess1,
                          '/home/wzh/ocr/Arithmetic_Func_detection_for_CTPN_v1/checkpoints/VGGnet_fast_rcnn_iter_25000.ckpt')




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

    return state, revise_output, output


def delete_top_or_bottom(label):

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

    if result == 'problem':                 #可能会把error变为problem
        result = result_temp

    return result,label_temp,label




def eval_label(label):
    if '=' not in label or label=='':
        return 'problem'
    else:
        left = label.split('=')[0]
        right = label.split('=')[1]

    if right=='' or left == '':
        return 'problem'

    left = left.replace('×', '*')
    if '÷' in left:
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
        result = eval(left)
        if result == int(right):
            return 'right'
        else:
            return 'error'





# label1 = '10-(3+5)=3'
# label2 = '10×5=51'
# label3 = '6÷3=2'
# label4 = '9÷4=2**1'
# label5 = '9÷4=2——1'
#
#
# a = eval_label(label4)
#
# print(a)




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('path',help='image path',type=str)
    # args = parser.parse_args()
    #img = cv2.imread(args.path)

    sess1, sess2, net, run_list, dense_decoder,inputs,width,is_training = create_sess()

    img_path = glob('/home/wzh/data/img/*')
    for path in tqdm(img_path):
        img = cv2.imread(path)
        save_path = path.replace('data/img','pipline_result/crnn1/big')


        result = pipline2(img.copy(),sess1, sess2, net, run_list, dense_decoder,inputs,width,is_training)
        x_pro = 3024 / img.shape[1]
        y_pro = 4031 / img.shape[0]
        img = cv2.resize(img, (3024, 4032))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # draw_box(img,result.right_error_result,x_pro,y_pro)
        # img = draw_result(img,result.right_error_result,x_pro,y_pro)
        # draw_box(img,result.problem_result,x_pro,y_pro)
        # img = draw_result(img,result.problem_result,x_pro,y_pro)

        draw_box(img, result.connect_result, x_pro, y_pro)
        img = draw_result(img, result.connect_result, x_pro, y_pro)

        # draw_box(img, result.results, x_pro, y_pro)
        # img = draw_result(img, result.results, x_pro, y_pro)

        img.save(save_path)

        # img.save('result12_1.jpg')
        # cv2.imshow("aa",img)
        # cv2.imwrite('./result2.JPG',img)
        # cv2.waitKey()