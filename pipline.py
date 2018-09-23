import utils
import tensorflow as tf
from model import CTC_Model
from config import Config as config
import cv2
from Arithmetic_Func_detection_for_CTPN_v1.ctpn import run
class Result(object):
    def __init__(self,bbox,img):
        self.top = bbox[1]
        self.bottom = bbox[3]
        self.left = bbox[0]
        self.right = bbox[2]
        self.img = img[self.top:self.bottom+1,self.left:self.right+1,...]
        self.normal_img = utils.image_normal(self.img.copy())
        self.img_wide = self.right+1-self.left
        self.output = ''
        self.state = 'right'

    def set_result(self,result):
        self.result = result


class All_Result(object):
    def __init__(self,bboxes,img):
        self.bboxes = bboxes
        self.img = img
        self.results = []
        self.max_wide = 0

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

def draw(img,all_result):
    for result in all_result:
        if result.state == 'right':
            rgb = (0,255,0)
        if result.state == 'error':
            rgb = (0,0,255)
        if result.state == 'problem':
            rgb = (255,0,0)
        cv2.rectangle(img,(result.left,result.top),(result.right,result.bottom),rgb)


def pipline(img):



    g1_config = tf.ConfigProto(allow_soft_placement=True)

    g1 = tf.Graph()
    sess1 = tf.Session(config = g1_config,graph=g1)

    with sess1.as_default():
        with g1.as_default():
            run_list,feed_dict, img_resized_shape, im_scales,bbox_scale = run.run_ctpn(img)
            saver = tf.train.Saver()
            saver.restore(sess1,
                          '/home/wzh/ocr/Arithmetic_Func_detection_for_CTPN_v1/checkpoints/VGGnet_fast_rcnn_iter_25000.ckpt')

            out_put = sess1.run(run_list, feed_dict)

            bboxes = run.decode_ctpn_output(out_put, im_scales,bbox_scale, img_resized_shape)


    # for bbox in bboxes:
    #     cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0))
    #
    # cv2.imshow('aaa',img)
    # cv2.waitKey()


    #
    all_result = All_Result(bboxes,img)


    image,wides = all_result.create_input()
    #

    #
    # for i ,result in enumerate(all_result.results):
    #     cv2.imwrite('/home/wzh/ocr/img/'+str(i)+'.jpg',result.img)
    #



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
            sentence = sess2.run(dense_decoder, feed_dict={inputs: image, width: wides, is_training: False})

            output = sentence.tolist()

            decode = dict(zip(config.ONE_HOT.values(), config.ONE_HOT.keys()))

            output = list(map(lambda y: ''.join(list(map(lambda x:decode.get(x), y))), output))

            for i,result in enumerate(all_result.results):
                result.output = output[i]
                try:
                    result.state = eval_label(output[i])
                except:
                    result.state = 'problem'

            print(output)


            return all_result



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

        if '*' in right or '——' in right:
            right1 = ''
            right2 = ''
            if '*' in right:
                right1 = right.split('*')[0]
                right2 = right.split('*')[-1]


            if '——' in right:
                right1 = right.split('——')[0]
                right2 = right.split('——')[-1]

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


# img = cv2.imread('IMG_5072.JPG')
# result = pipline(img.copy())
#
# draw(img,result.results)
# cv2.imshow("aa",img)
# cv2.waitKey()



# label1 = '10-(3+5)=3'
# label2 = '10×5=51'
# label3 = '6÷3=2'
label4 = '9÷4=2**1'
# label5 = '9÷4=2——1'
#
#
a = eval_label(label4)
#
print(a)

