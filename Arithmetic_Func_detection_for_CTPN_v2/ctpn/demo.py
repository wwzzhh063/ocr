from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):

    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])

    iiimg = cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return iiimg, f


def draw_boxes(img, boxes, color):
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[5])), color, 1)


def resize_bbox(boxes, scale):
    """
    将4点的坐标resize成相对原图的坐标，并返回左上和右下的2点坐标
    :param boxes:
    :param scale: 恢复原图尺寸坐标的缩放值
    :return: n*5, [x1,y1,x2,y2,score]
    """
    resized_bbox = []
    scores = []
    for box in boxes:
        bbox = []
        bbox.append(min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
        bbox.append(min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
        bbox.append(max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
        bbox.append(max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
        bbox.append(box[8])
        scores.append(box[8])
        resized_bbox.append(bbox)

    return resized_bbox, scores

def resize_pp(proposals, score, scale):
    resized_bbox = []
    scores = []
    index = 0
    for proposal in proposals:
        bbox = []
        bbox.append(int(proposal[0] / scale))
        bbox.append(int(proposal[1] / scale))
        bbox.append(int(proposal[2] / scale))
        bbox.append(int(proposal[3] / scale))
        bbox.append(score[index])
        resized_bbox.append(bbox)
        index += 1

    return resized_bbox


def output_bbox_txt(img_name, class_name, bboxes, save_dir='data/mAP/predicted/'):
    """
    输出预测bbox，并保存txt，保存格式为:class_name x1 y1 x2 y2
    :param img_name:
    :param class_name:
    :param bboxes:
    :param save_dir:
    :return:
    """
    img_id = img_name.split('.')[0]
    with open(os.path.join(save_dir, img_id+'.txt'), 'a+') as f:
        for bbox in bboxes:
            f.writelines(class_name)
            f.writelines(" ")
            f.writelines(str(bbox[4]))
            f.writelines(" ")
            f.writelines(str(bbox[0]))
            f.writelines(" ")
            f.writelines(str(bbox[1]))
            f.writelines(" ")
            f.writelines(str(bbox[2]))
            f.writelines(" ")
            f.writelines(str(bbox[3]))
            f.writelines("\n")


def ctpn(sess, net, image_path):
    timer = Timer()
    timer.tic()

    img_ = cv2.imread(image_path)
    img_name = image_path.split('/')[-1]
    # 将图像进行resize并返回其缩放大小
    img, scale = resize_im(img_, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # 送入网络得到1000个得分,1000个proposal
    cls, scores, boxes = test_ctpn(sess, net, img)


    detector = TextDetector()
    boxes, scores, cls = detector.proposal_nums(boxes, scores[:, np.newaxis], cls)

    img_re = img.copy()
    for i in range(np.shape(boxes)[0]):
        if cls[i] == 1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(img_re, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color, 1)
    cv2.imwrite(os.path.join('./data/proposal_res', img_name), img_re)

    # 获取手写proposal和其得分
    handwritten_filter = np.where(cls==1)[0]
    handwritten_scores = scores[handwritten_filter]
    handwritten_boxes = boxes[handwritten_filter, :]

    # 获取打印proposal和其得分
    print_filter = np.where(cls==2)[0]
    print_scores = scores[print_filter]
    print_boxes = boxes[print_filter, :]

    # print('print_filter', np.array(print_filter).shape)
    # print('handwritten_boxes, handwritten_scores', handwritten_boxes.shape, handwritten_scores[:, np.newaxis].shape)

    # 将手写和打印的proposal分别进行合并，得到bbox
    filted_handwritten_boxes = detector.detect(handwritten_boxes, handwritten_scores[:, np.newaxis], img.shape[:2])
    filted_print_boxes = detector.detect(print_boxes, print_scores[:, np.newaxis], img.shape[:2])

    res_handwritten_boxes, _ = resize_bbox(filted_handwritten_boxes, scale)
    res_print_boxes, _ = resize_bbox(filted_print_boxes, scale)

    draw_boxes(img, filted_handwritten_boxes, (255, 0, 0))
    draw_boxes(img, filted_print_boxes, (0, 255, 0))

    val_pp = False
    if val_pp:
        handwritten_boxes = resize_pp(handwritten_boxes, handwritten_scores, 1)
        print_boxes = resize_pp(print_boxes, print_scores, 1)
        output_bbox_txt(img_name, "handwritten", handwritten_boxes)
        output_bbox_txt(img_name, "print", print_boxes)
    else:
        output_bbox_txt(img_name, "handwritten", res_handwritten_boxes)
        output_bbox_txt(img_name, "print", res_print_boxes)


    img = cv2.resize(img, None, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", img_name), img)

    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    return timer.total_time


if __name__ == '__main__':

    if os.path.exists('data/mAP/predicted/'):
        shutil.rmtree("data/mAP/predicted/")
    os.makedirs("data/mAP/predicted/")

    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network 构建网络模型
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = test_ctpn(sess, net, im)
    test_path = os.path.join(cfg.DATA_DIR, 'demo_val', '*')
    # test_path = os.path.join('/home/tony/ocr/ocr_dataset/test/img', '*')
    im_names = glob.glob(test_path)
    # print(glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.JPG')))
    total_time = 0
    num = 0
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))

        time = ctpn(sess, net, im_name)
        if not time>=1:
            total_time += time
            num += 1


    print('avg time:', total_time/num)


"""
iou阈值=0.5
46.19% = handwritten AP  
58.49% = print AP  
mAP = 52.34%

iou阈值=0.4
68.55% = handwritten AP  
80.14% = print AP  
mAP = 74.35%

iou阈值=0.5
计算iou方式有变化，可以当做recall
86.37% = handwritten AP  
97.08% = print AP  
mAP = 91.73%

"""