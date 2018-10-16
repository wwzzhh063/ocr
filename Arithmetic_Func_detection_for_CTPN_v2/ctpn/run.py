from __future__ import print_function

import cv2
import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf


from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import run
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):

    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])

    iiimg = cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return iiimg, f

def resize_bbox(boxes, scale):
    resized_bbox = []
    scores = []
    for box in boxes:
        bbox = []
        bbox.append(min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
        bbox.append(min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
        bbox.append(max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
        bbox.append(max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
        scores.append(box[8])
        resized_bbox.append(bbox)

    return resized_bbox, scores

def build_ctpn_model():

    path = os.path.abspath(os.path.curdir)

    if 'Arithmetic_Func_detection_for_CTPN' not in path:
        path = os.path.join(path, 'Arithmetic_Func_detection_for_CTPN_v2/ctpn/text.yml')
    else:
        path = path + '/ctpn/text.yml'
        print(path)
    cfg_from_file(path)
    config = tf.ConfigProto(allow_soft_placement=True)
    # load network 构建网络模型
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    run_list = [net.get_output('rois')[0], net.get_output('rpn_targets')]
    return net, run_list

def run_ctpn(img, net):

    # 　将图像进行resize并返回其缩放大小
    img_resized, bbox_scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # print(scale)
    run_list, feed_dict, im_scales = run(net, img_resized)

    return feed_dict, img_resized.shape, im_scales, bbox_scale

def decode_ctpn_output(ctpn_output, im_scales, bbox_scale, img_resized_shape):
    rois = ctpn_output[0]

    cls = rois[:, 0]
    scores = rois[:, 1]
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 2:6] / im_scales[0]

    textdetector = TextDetector()
    boxes, scores, cls = textdetector.proposal_nums(boxes, scores[:, np.newaxis], cls)

    # 获取手写proposal和其得分
    handwritten_filter = np.where(cls == 1)[0]
    handwritten_scores = scores[handwritten_filter]
    handwritten_boxes = boxes[handwritten_filter, :]

    # 获取打印proposal和其得分
    print_filter = np.where(cls == 2)[0]
    print_scores = scores[print_filter]
    print_boxes = boxes[print_filter, :]

    filted_handwritten_boxes = textdetector.detect(handwritten_boxes, handwritten_scores[:, np.newaxis], img_resized_shape[:2])
    filted_print_boxes = textdetector.detect(print_boxes, print_scores[:, np.newaxis], img_resized_shape[:2])

    res_handwritten_boxes, _ = resize_bbox(filted_handwritten_boxes, bbox_scale)
    res_print_boxes, _ = resize_bbox(filted_print_boxes, bbox_scale)


    label = [1 for _ in range(len(res_handwritten_boxes))]
    label.extend([2 for _ in range(len(res_print_boxes))])

    res_handwritten_boxes.extend(res_print_boxes)

    return res_handwritten_boxes, label


def main():
    img = cv2.imread(
        '/home/tony/ocr/Arithmetic_Func_detection_for_CTPN/data/demo_test/1B718AD0-513E-4593-9623-EC6B5AEB792C.png')
    net, run_list = build_ctpn_model()

    sess = tf.Session()
    saver = tf.train.Saver()

    feed_dict, img_resized_shape, im_scales, bbox_scale = run_ctpn(img, net)

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    out_put = sess.run(run_list, feed_dict)

    all_bboxes, label = decode_ctpn_output(out_put, im_scales, bbox_scale, img_resized_shape)

    # print(all_bboxes)
    print(len(all_bboxes), len(label))
    for bbox in all_bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    cv2.imwrite('dwad.jpg', img)

if __name__ == "__main__":
    main()