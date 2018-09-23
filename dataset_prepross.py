from glob import glob
import os
import config
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import cutdata
path = '/home/wzh/prepocess1'
if not os.path.exists(path):
    os.mkdir(path)

def enhance(img):
    if len(img.shape) == 2:
        sort_img = np.sort(np.ndarray.flatten(img))
        avager = np.min(sort_img[int(sort_img.shape[0] * 0.75):])
    else:
        sort_img = np.sort(np.reshape(img, [-1, 3]), axis=0)
        avager = np.min(sort_img[int(sort_img.shape[0] * 0.75):, :], axis=0)

    c_out = np.minimum(np.ones(img.shape), img / avager)
    mask = 0.5 - 0.5 * np.cos(0.75 * c_out * np.pi)
    return mask

def preprocess(img,img2,img2_,size = 15):

    img2 = img2.copy()
    mask = np.zeros(img.shape)

    i = 0
    while i <= img.shape[0]:
        if i + size > img.shape[0]:
            i_end = img.shape[0]
        else:
            i_end = i + size
        j = 0

        while j <= img.shape[1]:
            if j + size > img.shape[1]:
                j_end = img.shape[1]
            else:
                j_end = j + size

            part_mask = enhance(img[i:i_end, j:j_end])
            mask[i:i_end, j:j_end] = part_mask
            j = j + size

        i = i + size

    img1 = np.ones(img.shape) * mask * 255
    #cv2.imwrite("./img1.jpg",img1)
    img2 = 255-img2
    img2_ = 255 - img2_
    #cv2.imwrite("./img2.jpg", img2)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    img2_ = cv2.cvtColor(img2_,cv2.COLOR_GRAY2BGR)
    img3 = img1*(img2/255)
    img4 = img1**(img2_/255)
    # cv2.imwrite("./img3.jpg", img3)
    return img3,img4


def img_normal(img,img_path):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img,(3,3))

    canny = cv2.Canny(img,25,50,apertureSize=3)

    gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)              #水平方向sobel算子
    gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)              #垂直方向sobel算子

    # subtract the y-gradient from the x-gradient
    gradient = cv2.add(np.abs(gradX), np.abs(gradY))                                 #平滑
    (_, thresh) = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)
    # thresh = 255-thresh

    cv2.imwrite(img_path.replace('.', '_1.'), thresh)
    cv2.imwrite(img_path.replace('.', '_2.'), canny)

    return thresh,canny

for i in tqdm(glob('/home/wzh/img/*')):
    try:
        img = cv2.imread(i)
        img1,img2 = img_normal(img.copy(),i)
        img3,img4 = preprocess(img.copy(),img1,img2)
        cv2.imwrite(i.replace('.', '_3.'), img3)
        cv2.imwrite(i.replace('.', '_4.'), img4)
    except:
        print(i)
