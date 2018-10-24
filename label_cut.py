from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
from glob import glob
import cv2
import argparse
from PIL import Image,ImageDraw,ImageFont
import math
def get_pair_by_distance(print_word_all,hand_word_all):
    # all_bbox = Img_ALL_BBox()

    print_hand = {}
    hand_print = {}
    for i,print_word in enumerate(print_word_all):         #手写到打印匹配一遍
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(hand_word_all):               #算距离
            distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
            if min_distance>distance:
                pair = j
                min_distance = distance

        try:
            if in_same_line(print_word.bbox,hand_word_all[pair].bbox) == 'in':  #算是否在一行
                print_hand[i] = pair
                if hand_print.get(pair):
                    hand_print[pair].append(i)
                else:
                    hand_print[pair] = [i]
        except:
            pass


    for key in hand_print:                      #打印到手写再匹配一遍
        if len(hand_print[key])>1:
            min_distance = 9999
            min_value = -1
            for print in hand_print[key]:
                print_word = print_word_all[print]
                hand_word = hand_word_all[key]
                distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
                if min_distance>distance:
                    min_distance = distance
                    print_hand.pop(min_value,'none')
                    min_value = print
                else:
                    print_hand.pop(print)


    return print_hand,hand_print

def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))


def bbbox_to_distance(point_i,point_j):
    if len(point_i) == 4:
        print_word_point = (point_i[2], (point_i[1] + point_i[3]) / 2)
    else:
        print_word_point = ((point_i[2] + point_i[4]) / 2, (point_i[3] + point_i[5]) / 2)

    if len(point_j) == 4:
        hand_word_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
    else:
        hand_word_ponit = ((point_j[0] + point_j[6]) / 2, (point_j[1] + point_j[7]) / 2)

    distences = get_distance(hand_word_ponit, print_word_point)

    return distences



def to_int(a):
    x,y = a
    return (int(x),int(y))

def in_same_line(print_bbox,hand_bbox):
    if len(print_bbox) == 4:
        centre_print_bbox = (print_bbox[1] + print_bbox[3]) / 2
    else:
        centre_print_bbox =  (print_bbox[3] + print_bbox[5]) / 2

    if len(hand_bbox) == 4:
        if centre_print_bbox>hand_bbox[1] and centre_print_bbox<hand_bbox[3]:
            return 'in'
        else:
            return 'out'
    else:
        if centre_print_bbox > hand_bbox[1] and centre_print_bbox < hand_bbox[7]:
            return 'in'
        else:
            return 'out'

def get_pair_by_distance2(print_word_all,hand_word_all):
    # all_bbox = Img_ALL_BBox()

    print_hand = {}
    hand_print = {}
    for i,print_word in enumerate(print_word_all):         #手写到打印匹配一遍
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(hand_word_all):               #算距离
            distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
            if min_distance>distance:
                pair = j
                min_distance = distance

        try:
            if in_same_line(print_word.bbox,hand_word_all[pair].bbox) == 'in' and min_distance<(print_word.bbox[2]-print_word.bbox[0])/3 :  #算是否在一行
                print_hand[i] = pair
                if hand_print.get(pair):
                    hand_print[pair].append(i)
                else:
                    hand_print[pair] = [i]
        except:
            pass


    for key in hand_print:                      #打印到手写再匹配一遍
        if len(hand_print[key])>1:
            min_distance = 9999
            min_value = -1
            for print in hand_print[key]:
                print_word = print_word_all[print]
                hand_word = hand_word_all[key]
                distance = bbbox_to_distance(print_word.bbox,hand_word.bbox)
                if min_distance>distance:
                    min_distance = distance
                    print_hand.pop(min_value,'none')
                    min_value = print
                else:
                    print_hand.pop(print)


    return print_hand,hand_print




class Bbox(object):
    def __init__(self,bbox,label,classes):
        self.bbox = bbox
        self.label = label
        self.classes = classes
        self.state = 'start'
        self.type = '='

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



class Img_ALL_BBox(object):
    def __init__(self,img):
        self.print_word = []
        self.hand_word = []
        self.merge = []
        self.img_path = ''
        self.pair = {}
        self.img = img
        self.not_pair = []
        self.all_box = []
        self.problem_box = []

    def cut_biger_img(self,path,num=0):
        for i,print_num in enumerate(self.pair):
            hand_num = self.pair[print_num]
            print = self.print_word[print_num]
            hand = self.hand_word[hand_num]
            top = min(print.top,hand.top)
            bottom = max(print.bottom,hand.bottom)
            left = min(print.left,hand.left)
            right = max(print.right,hand.right)

            biger_img = self.img[top:bottom+1,left:right+1,...]

            label = print.label+hand.label
            if 'None' not in label:                     #test----------------------------------------
                cv2.imwrite(path+str(num)+'_'+str(i)+'_'+label+'.jpg',biger_img)

    def create_big_img(self,pair,box_list1,box_list2):        #先合并非括号填词
        box_list1_ = box_list1.copy()
        box_list2_ = box_list2.copy()
        try:
            for i, print_num in enumerate(pair):
                hand_num = pair[print_num]
                print = box_list1_[print_num]
                hand = box_list2_[hand_num]
                box_list1.remove(print)
                box_list2.remove(hand)
                top = min(print.top, hand.top)
                bottom = max(print.bottom, hand.bottom)
                left = min(print.left, hand.left)
                right = max(print.right, hand.right)


                label = print.label + hand.label
                big_img = Bbox([left,top,right,bottom],label,'merge')
                if '*' in big_img.label or '~' in big_img.label:
                    big_img.type = '...'
                self.merge.append(big_img)
        except:
            print('a')

        self.not_pair = box_list1+box_list2


    def create_big_img2(self,pair,box_list1,box_list2):         #合并括号填词的
        box_list1_ = box_list1.copy()
        box_list2_ = box_list2.copy()
        try:
            for i, print_num in enumerate(pair):
                hand_num = pair[print_num]
                print = box_list1_[print_num]
                hand = box_list2_[hand_num]
                box_list1.remove(print)
                box_list2.remove(hand)
                top = min(print.top, hand.top)
                bottom = max(print.bottom, hand.bottom)
                left = min(print.left, hand.left)
                right = max(print.right, hand.right)


                label = print.label + hand.label
                big_img = Bbox([left,top,right,bottom],label,'merge')
                big_img.type = '()'
                self.merge.append(big_img)
        except:
            print('a')

        self.not_pair = box_list2
        self.all_box = self.merge+self.not_pair

    def create_pair(self):
        print_hand, hand_print = get_pair_by_distance(self.print_word, self.hand_word)
        self.pair = print_hand
        self.create_big_img(self.pair, self.print_word, self.hand_word)
        pair, _ = get_pair_by_distance2(self.merge, self.not_pair)
        self.create_big_img2(pair, self.merge, self.not_pair)




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


def find_labels(name,j,label_path):
    name = name.split('.')[0]
    name = name.split('_')[-1]
    path = label_path
    paths = glob(os.path.join(path,'*'))
    label_path = ''
    for i in paths:
        # if name == '201809041126441' :
        #     print('a')
        if name in i:
            temp = i.split(name)[-1][0]
            if  temp not in ['0','1','2','3','4','5','6','7','8','9']:
                label_path = i
                j = j+1
                break
    return j,label_path

def label_replace(label):
    label = label.replace('（','(')
    label = label.replace('）',')')
    label = label.replace('４','4')
    label = label.replace('１','1')
    label = label.replace('５','5')
    label = label.replace('８','8')
    label = label.replace('９','9')
    label = label.replace('＋','+')
    label = label.replace('２','2')
    label = label.replace('０','0')
    label = label.replace('６','6')
    label = label.replace('３','3')
    label = label.replace('７','7')
    label = label.replace('－','-')
    label = label.replace('　','')
    label = label.replace('？','?')
    label = label.replace('，',',')
    label = label.replace('：',':')
    label = label.replace('＞','>')
    label = label.replace('！','!')
    label = label.replace('＝','=')
    label = label.replace('—','~')
    label = label.replace('√','')
    label = label.replace(' ','')
    label = label.replace('＇',"'")




    label = label.replace('①', 'None')
    label = label.replace('②', 'None')
    label = label.replace('③', 'None')
    label = label.replace('④', 'None')
    label = label.replace('_','')
    label = label.replace('一','1')
    label = label.replace('二', '2')
    label = label.replace('五', '5')
    label = label.replace('/','')

    return label

list = []
def read_label(xml):
    try:
        tree = ET.parse(xml)
    except:
        print(xml)
    root = tree.getroot()
    label = root.find('outputs/transcript')

    if label != None:
        label = label.text
        label = label_replace(label)
        if label != 'None' and label != 'Good':
            for j in label:
                if j not in list:
                    list.append(j)

    else:
        label = 'None'

    name = root.find('path').text
    name = name.split('\\')[-1]

    return label,name



def set_xml_data(path):
    xml_path = glob(os.path.join(path, '*.xml'))
    img_name = []           #所有图片名称
    all_img_label  = []             #所有图片的标签
    img_path = path.replace('xml','img')

    j = 0
    for big_img_xml in tqdm(xml_path):            #所有xml
        p = ParseXml(big_img_xml)
        img_name_, class_list, bbox_list,jpg_or_JPG = p.get_bbox_class()
        big_img_path = os.path.join(img_path,big_img_xml.split('/')[-1].replace('xml',jpg_or_JPG))
        all_bbox_img = Img_ALL_BBox(cv2.imread(big_img_path))           #一张图片中的所有box   tetst---------------------------------
        all_bbox_img.img_path = big_img_path
        img_name.append(img_name_)
          #得到一张图片所有的xml
        for i, classes in enumerate(class_list):  # 一张图片中每一个box
            bbox = Bbox(bbox_list[i], '?', classes)
            # bbox = Bbox_test(bbox_list[i], label, classes,small_img_xml_path)          #for test!!!!!!!!!!
            if classes == 1:
                all_bbox_img.print_word.append(bbox)
            else:
                all_bbox_img.hand_word.append(bbox)

        all_img_label.append(all_bbox_img)

    return all_img_label




def draw_bbox(bbox,img,x_pro,y_pro,color):
    cv2.rectangle(img, (int(bbox[0] * x_pro), int(bbox[1] * y_pro)),
                  (int(bbox[2] * x_pro), int(bbox[3] * y_pro)), color, 4)

def draw_result(img,label,x_pro,y_pro):
    ttfont = ImageFont.truetype('SimSun.ttf',25)
    draw = ImageDraw.Draw(img)
    draw.text((int(label.left*x_pro),int(label.bottom*y_pro)),label.label,fill='blue',font=ttfont)
        # cv2.putText(img,result.output,(result.left,result.top),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    return img




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_path', help='xml path', type=str)
    parser.add_argument('save_path', help='label path', type=str)
    args = parser.parse_args()
    all_img_label = set_xml_data(args.xml_path)
    for img_label in tqdm(all_img_label):
        img_label.create_pair()
        img = cv2.imread(img_label.img_path)
        # x_pro = 3024 / img.shape[1]
        # y_pro = 4031 / img.shape[0]
        # img = cv2.resize(img, (3024, 4032))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        path = os.path.join(args.save_path, img_label.img_path.split('/')[-1].split('.')[0])
        if not os.path.exists(path):
            os.mkdir(path)
        for i,label in enumerate(img_label.all_box):
            cut_img = img[label.top:label.bottom+1,label.left:label.right+1,...]
            name = str(i)+'_'+str(label.bbox)+'.jpg'
            cv2.imwrite(os.path.join(path,name),cut_img)


