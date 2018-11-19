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

    canny = cv2.blur(canny, (3, 3))

    gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)              #水平方向sobel算子
    gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)              #垂直方向sobel算子

    # subtract the y-gradient from the x-gradient
    gradient = cv2.add(np.abs(gradX), np.abs(gradY))                                 #平滑
    (_, thresh) = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)
    # thresh = 255-thresh

    # cv2.imwrite(img_path.replace('.', '_1.'), thresh)
    # cv2.imwrite(img_path.replace('.', '_2.'), canny)

    return thresh,canny

def write_img():
    for i in tqdm(glob('/home/wzh/img/*')):
        try:
            img = cv2.imread(i)
            img1,img2 = img_normal(img.copy(),i)
            img3,img4 = preprocess(img.copy(),img1,img2)
            cv2.imwrite(i.replace('.', '_3.'), img3)
            cv2.imwrite(i.replace('.', '_4.'), img4)
        except:
            print(i)

def hough(img,img2):
    h,w = img.shape
    centre_y,centre_x = h/2,w/2

    # cv2.imwrite("b.jpg",img)
    img = np.asarray(img,np.uint8)
    lines = cv2.HoughLines(img,1,np.pi/180,750)
    lines2 = cv2.HoughLinesP(img,1,np.pi/180,80,200,15)

    for i ,line in tqdm(enumerate(lines2)):
        print_line(line[0],img2)

    cv2.imwrite("a.jpg",img2)
    print("a")

def print_line(line,img):
    rho = line[0]  # 第一个元素是距离rho
    theta = line[1]  # 第二个元素是角度theta
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
        # 该直线与第一行的交点
        pt1 = (int(rho / np.cos(theta)), 0)
        # 该直线与最后一行的焦点
        pt2 = (int((rho - img.shape[0] * np.sin(theta)) / np.cos(theta)), img.shape[0])
        # 绘制一条白线
        cv2.line(img, pt1, pt2, (0,0,255),2)
    else:  # 水平直线
        # 该直线与第一列的交点
        pt1 = (0, int(rho / np.sin(theta)))
        # 该直线与最后一列的交点
        pt2 = (img.shape[1], int((rho - img.shape[1] * np.cos(theta)) / np.sin(theta)))
        # 绘制一条直线
        cv2.line(img, pt1, pt2, (0,0,255), 2)

def texiao(img_path,save_path,divide = 100):
    img_orginal = cv2.imread(img_path)
    img = cv2.cvtColor(img_orginal, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (3, 3))

    gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)  # 水平方向sobel算子
    gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)  # 垂直方向sobel算子

    # subtract the y-gradient from the x-gradient
    gradient = cv2.add(np.abs(gradX), np.abs(gradY))  # 平滑
    (_, thresh) = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)

    canny = cv2.Canny(img, 25, 50, apertureSize=3)
    canny = (255-canny)/255
    canny = (255-thresh)/255

    lenght = img.shape[0]
    ever_lenght = int(lenght/divide)
    for i in tqdm(range(divide)):
        if i+1 == divide:
            end = lenght
        else:
            end = (i+1)*ever_lenght
        canny_temp = np.ones(canny.shape)
        canny_temp[i*ever_lenght:end] = canny[i*ever_lenght:end]
        img_temp = img_orginal*canny_temp[:,:,np.newaxis]
        color = np.ones(img_orginal.shape)*(1-canny_temp[:,:,np.newaxis])*(0,255,0)
        img_temp = img_temp+color
        cv2.imwrite(os.path.join(save_path,str(i)+'.jpg'),img_temp)

def dection_circles(img_h,img_o):
    img_h = np.asarray(img_h,np.uint8)
    circle1 = cv2.HoughCircles(img_h, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=50, minRadius=0, maxRadius=500)
    circles = circle1[0, :, :]  # 提取为二维
    circles = np.uint16(np.around(circles))  # 四舍五入，取整
    for i in circles[:]:
        cv2.circle(img_o, (i[0], i[1]), i[2], (0, 0, 0), 5)

    return img_o



path = '/home/wzh/ocr/lADPDgQ9qVqmuZLNAWPNAkM_579_355.jpg'
img = cv2.imread(path)
img1, img2 = img_normal(img.copy(), path)
img_1o = dection_circles(img1,img.copy())
img_2o = dection_circles(img2,img.copy())

cv2.imwrite(path.replace('.', '_1.'), img1)
cv2.imwrite(path.replace('.', '_3.'), img2)

cv2.imwrite(path.replace('.', '_2.'), img_1o)
cv2.imwrite(path.replace('.', '_4.'), img_2o)


# img3, img4 = preprocess(img.copy(), img1, img2)
# cv2.imwrite(path.replace('.', '_3.'), img3)
# cv2.imwrite(path.replace('.', '_4.'), img4)

# texiao('hs(8).jpg','texiao')




