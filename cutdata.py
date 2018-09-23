import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

class ParseXml(object):

    def __init__(self, xml_path, rect=False):
        self.classes = []
        self.bbox = []
        self.rect = rect
        self.img_name = xml_path.split('/')[-1].replace('.xml', '')
        # print(self.img_name)
        self.res = self._read_xml(xml_path)

    def get_bbox_class(self):

        if self.res is True:
            return self.img_name, self.classes, self.bbox,self.jpg_or_JPG
        else:
            return self.img_name, None, None

    def _read_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        itmes = root.findall("outputs/object/item")

        self.jpg_or_JPG = root.find('path').text.split('.')[-1]

        for i in itmes:
            res = self._parse_item(i)
            if res is False:
                return False
        return True

    def _parse_item(self, item):
        class_elem = item.find('name')



        if item.find('bndbox'):
            bbox = []
            bndbox = item.find('bndbox')


            bbox.append(int(bndbox.find('xmin').text))
            bbox.append(int(bndbox.find('ymin').text))
            bbox.append(int(bndbox.find('xmax').text))
            bbox.append(int(bndbox.find('ymax').text))
            self.bbox.append(bbox)
            self.classes.append(int(class_elem.text))
            return True
        elif item.find('polygon'):
            pos = []
            polygon = item.find('polygon')
            pos.append(int(polygon.find('x1').text))
            pos.append(int(polygon.find('y1').text))

            if polygon.find('x2') is not None:
                pos.append(int(polygon.find('x2').text))
                pos.append(int(polygon.find('y2').text))
            else:
                print('img error:', self.img_name)
                print('多边形框选有问题,少点')
                return False

            pos.append(int(polygon.find('x3').text))
            pos.append(int(polygon.find('y3').text))

            if polygon.find('y4') is not None:
                pos.append(int(polygon.find('x4').text))
                pos.append(int(polygon.find('y4').text))

                if not self.rect:
                    self.bbox.append(pos)
                else:
                    bbox = []
                    bbox.append(min(pos[0],pos[2],pos[4],pos[6]))
                    bbox.append(min(pos[1], pos[3], pos[5], pos[7]))
                    bbox.append(max(pos[0], pos[2], pos[4], pos[6]))
                    bbox.append(max(pos[1], pos[3], pos[5], pos[7]))
                    self.bbox.append(bbox)
                self.classes.append(int(class_elem.text))
            else:
                print('img error:', self.img_name)
                print('多边形框选有问题,少点')
                return False

            if polygon.find('x5'):
                print('img error:', self.img_name)
                print('多边形框选有问题.多点')
                return False

            return True
        else:
            print('img error:', self.img_name)
            print('含有其他类型bbox')
            return False


def draw_bbox(img_name, class_list, bbox_list):
    img_name = img_name.replace('xml', 'JPG')
    img_path = '/home/wzh/ocr-demo/pic/'
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    img = cv2.imread(img_path + img_name)

    print_color = (0, 0, 255)
    hand_color = (255, 0, 0)

    for i in range(len(class_list)):
        if class_list[i] == 0:
            color = print_color
        else:
            color = hand_color

        if len(bbox_list[i]) == 4:
            cv2.rectangle(img, (bbox_list[i][0], bbox_list[i][1]), (bbox_list[i][2], bbox_list[i][3]), color, 2)
        else:
            cv2.line(img, (bbox_list[i][0], bbox_list[i][1]),
                     (bbox_list[i][2], bbox_list[i][3]), color, 2)
            cv2.line(img, (bbox_list[i][2], bbox_list[i][3]),
                     (bbox_list[i][4], bbox_list[i][5]), color, 2)
            cv2.line(img, (bbox_list[i][4], bbox_list[i][5]),
                     (bbox_list[i][6], bbox_list[i][7]), color, 2)
            cv2.line(img, (bbox_list[i][6], bbox_list[i][7]),
                     (bbox_list[i][0], bbox_list[i][1]), color, 2)
    cv2.imwrite('/home/wzh/ocr-demo/draw/' + img_name, img)



def cut_img(img_name, bbox_list):
    img_name = img_name.replace('xml', 'JPG')
    img_path = os.path.join('/home/wzh/╩╘╠т╩╓╨┤╠х▒ъ╫в╜с╣√/IMG',img_name)
    img = cv2.imread(img_path)
    for i,box in enumerate(bbox_list):
        if len(box) == 4:
            x_min = box[0]
            x_max = box[2]
            y_min = box[1]
            y_max = box[3]
            # x_expand = int((x_max-x_min)/50)
            # y_expand = int((y_max-y_min)/20)
        else:
            x_min = min(box[0],box[6])
            x_max = max(box[2],box[4])
            y_min = min(box[1],box[3])
            y_max = max (box[7],box[5])
            # x_expand = int((x_max - x_min) / 50)
            # y_expand = int((y_max - y_min) / 20)

        cut_img = img[y_min:y_max+1,x_min:x_max+1,:]
        # cut_img = img[y_min-y_expand:y_max + y_expand, x_min-x_expand:x_max + x_expand, :]


        path = os.path.join('/home/wzh/ocr_img',img_name.replace('.JPG',''))
        # path = os.path.join('/home/wzh/ocr_img', img_name.replace('.JPG', '')+'_expand')
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path,str(i)+'.jpg'),cut_img)




def img_normal(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img,(3,3))

    canny = cv2.Canny(img,25,50,apertureSize=3)

    gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)              #水平方向sobel算子
    gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)              #垂直方向sobel算子

    # subtract the y-gradient from the x-gradient
    gradient = cv2.add(np.abs(gradX), np.abs(gradY))                                 #平滑
    (_, thresh) = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)
    # thresh = 255-thresh

    cv2.imwrite(img_path.replace('.', '_6.'), thresh)
    cv2.imwrite(img_path.replace('.', 'canny_6.'), canny)

    return canny



def hough(img):
    h,w = img.shape
    centre_y,centre_x = h/2,w/2

    cv2.imwrite("b.jpg",img)
    img = np.asarray(img,np.uint8)
    lines = cv2.HoughLines(img,1,np.pi/180,750)
    lines2 = cv2.HoughLinesP(img,1,np.pi/180,80,200,15)

    for i ,line in tqdm(enumerate(lines)):
        print_line(line[0],img)

    cv2.imwrite("a.jpg",img)
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
        cv2.line(img, pt1, pt2, (255))
    else:  # 水平直线
        # 该直线与第一列的交点
        pt1 = (0, int(rho / np.sin(theta)))
        # 该直线与最后一列的交点
        pt2 = (img.shape[1], int((rho - img.shape[1] * np.cos(theta)) / np.sin(theta)))
        # 绘制一条直线
        cv2.line(img, pt1, pt2, (255), 1)


def globe_enhance(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img.jpg', img)
    mask = enhance(img)
    img1 = img*mask
    cv2.imwrite('img1.jpg',img1)


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



def local_enhance(img_path,size=60):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img.jpg', img)
    mask = np.zeros(img.shape)

    i = 0
    while i <=img.shape[0]:
        if i+size>img.shape[0]:
            i_end= img.shape[0]
        else:
            i_end = i+size
        j = 0

        while j<=img.shape[1]:
            if j+size>img.shape[1]:
                j_end = img.shape[1]
            else:
                j_end = j+size

            part_mask = enhance(img[i:i_end,j:j_end])
            mask[i:i_end,j:j_end] = part_mask
            j = j+size

        i = i+size

    img1 = np.ones(img.shape)*mask*255
    cv2.imwrite('img'+str(size)+'.jpg',img1)

def img_normal_(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img,(3,3))

    canny = cv2.Canny(img,25,50,apertureSize=3)

    gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)              #水平方向sobel算子
    gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)              #垂直方向sobel算子

    # subtract the y-gradient from the x-gradient
    gradient = cv2.add(np.abs(gradX), np.abs(gradY))                                 #平滑
    (_, thresh) = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)
    # thresh = 255-thresh

    return canny

def preprocess(img,size = 15):

    img2 = img_normal_(img.copy())
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
    #cv2.imwrite("./img2.jpg", img2)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    img3 = img1*(img2/255)
    cv2.imwrite("./img3.jpg", img3)
    return img3






if __name__ =="__main__":

    xml_path = '/home/wzh/ocr-demo/outputs/'
    xml_name = os.listdir(xml_path)
    # local_enhance('2.jpeg')
    # hough(img_normal('2.jpeg'))
    # img_normal('2.jpeg')
    preprocess(cv2.imread("１.JPG"))

    # for i in tqdm(range(len(xml_name))):
    #     name = xml_path+xml_name[i]
    #     p = ParseXml(name)
    #     img_name, class_list, bbox_list = p.get_bbox_class()
    #     draw_bbox( img_name, class_list, bbox_list)
        # cut_img(img_name,bbox_list)
        # img_enhance(img_name)

    # a = np.arange(0,9).reshape([3, 3])
    # print(a)
    # b = a[:, 0::3]
    # print(b)