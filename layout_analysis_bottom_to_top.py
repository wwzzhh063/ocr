from inference import set_xml_data,draw_bbox
from config import Config as config
import cv2
from tqdm import tqdm
import math
import random

#Python3.6
class Point(): #定义类
    def __init__(self,x,y):
        self.x=x
        self.y=y

def cross(p1,p2,p3):#跨立实验
    x1=p2.x-p1.x
    y1=p2.y-p1.y
    x2=p3.x-p1.x
    y2=p3.y-p1.y
    return x1*y2-x2*y1

def IsIntersec(p1,p2,p3,p4): #判断两线段是否相交

    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1.x,p2.x)>=min(p3.x,p4.x)    #矩形1最右端大于矩形2最左端
    and max(p3.x,p4.x)>=min(p1.x,p2.x)   #矩形2最右端大于矩形最左端
    and max(p1.y,p2.y)>=min(p3.y,p4.y)   #矩形1最高端大于矩形最低端
    and max(p3.y,p4.y)>=min(p1.y,p2.y)): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
           and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            D=1
        else:
            D=0
    else:
        D=0
    return D








class Layout_Cell(object):
    def __init__(self,bbox,label='',type=''):
        self.bbox = bbox
        self.label = label
        self.type = type
        self.row = -1
        self.Column = -1
        self.centre = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))

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


def row_iou(row1,row2):
    max_top = max(row1[0],row2[0])
    min_bottom = min(row1[1],row2[1])

    if max_top>=min_bottom:
        return 0
    else:
        return (min_bottom-max_top)/(row1[1]-row1[0]+row2[1]-row2[0]+max_top-min_bottom)

def column_iou(column1,column2):
    max_left = max(column1[0],column2[0])
    min_right = min(column1[2],column2[2])

    if max_left>=min_right:
        return 0
    else:
        try:
            return (min_right-max_left)/(column1[2]-column1[0]+column2[2]-column2[0]-min_right+max_left)
        except:
            print('a')

def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))



def in_same_line(print_cell_bbox,hand_bbox):
    if len(print_cell_bbox) == 4:
        centre_print_cell_bbox = (print_cell_bbox[1] + print_cell_bbox[3]) / 2
    else:
        centre_print_cell_bbox =  (print_cell_bbox[3] + print_cell_bbox[5]) / 2

    if len(hand_bbox) == 4:
        if centre_print_cell_bbox>hand_bbox[1] and centre_print_cell_bbox<hand_bbox[3]:
            return 'in'
        else:
            return 'out'
    else:
        if centre_print_cell_bbox > hand_bbox[1] and centre_print_cell_bbox < hand_bbox[7]:
            return 'in'
        else:
            return 'out'

def row_get_pair_by_distance(print_cell_word_all,hand_word_all):

    def bbbox_to_distance(point_i, point_j):
        if len(point_i) == 4:
            print_cell_word_point = (point_i[2], (point_i[1] + point_i[3]) / 2)
        else:
            print_cell_word_point = ((point_i[2] + point_i[4]) / 2, (point_i[3] + point_i[5]) / 2)

        if len(point_j) == 4:
            hand_word_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
        else:
            hand_word_ponit = ((point_j[0] + point_j[6]) / 2, (point_j[1] + point_j[7]) / 2)

        distences = get_distance(hand_word_ponit, print_cell_word_point)

        return distences


    print_cell_hand = {}
    hand_print_cell = {}
    for i,print_cell_word in enumerate(print_cell_word_all):         #手写到打印匹配一遍
        min_distance = 9999
        pair = -1
        for j,hand_word in enumerate(hand_word_all):               #算距离
            distance = bbbox_to_distance(print_cell_word.bbox,hand_word.bbox)
            if min_distance>distance :
                pair = j
                min_distance = distance

        try:
            if in_same_line(print_cell_word.bbox,hand_word_all[pair].bbox) == 'in' and min_distance<(print_cell_word.bbox[2]-print_cell_word.bbox[0])/3.5 \
                    and column_iou(print_cell_word.bbox,hand_word_all[pair].bbox)<0.1:  #算是否在一行
                print_cell_hand[i] = pair
                if hand_print_cell.get(pair):
                    hand_print_cell[pair].append(i)
                else:
                    hand_print_cell[pair] = [i]
        except:
            pass


    for key in hand_print_cell:                      #打印到手写再匹配一遍
        if len(hand_print_cell[key])>1:
            min_distance = 9999
            min_value = -1
            for print_cell in hand_print_cell[key]:
                print_cell_word = print_cell_word_all[print_cell]
                hand_word = hand_word_all[key]
                distance = bbbox_to_distance(print_cell_word.bbox,hand_word.bbox)
                if min_distance>distance:
                    min_distance = distance
                    print_cell_hand.pop(min_value,'none')
                    min_value = print_cell
                else:
                    print_cell_hand.pop(print_cell)


    return print_cell_hand




def column_get_pair_by_distance(boxes):



    def bbbox_to_distance(point_i, point_j):
        if len(point_i) == 4:
            print_cell_word_point = (point_i[0], (point_i[1] + point_i[3]) / 2)
        else:
            pass

        if len(point_j) == 4:
            hand_word_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
        else:
            pass

        distences = get_distance(hand_word_ponit, print_cell_word_point)

        return distences



    box_top_to_bottom = {}
    box_bottom_to_top = {}

    for i,box_top in enumerate(boxes):
        min_distance = 9999
        pair = -1
        min_top = 9999
        for j,box_bottom in enumerate(boxes):
            # if i == 12 and j == 22:
            #     print('a')

            if box_top is box_bottom:
                continue
            else:
                distnace = bbbox_to_distance(box_top.bbox,box_bottom.bbox)
                top = box_bottom.top
                # distnace = box_bottom.top - box_top.top
                if box_top.top>box_bottom.top:
                    continue
                if  distnace<min_distance and column_iou(box_top.bbox,box_bottom.bbox)>0.1 \
                        and ((distnace < (box_top.bottom-box_top.top)*4 or distnace < (box_bottom.bottom-box_bottom.top)*4))  \
                        or (distnace<min_distance and distnace < (box_top.bottom-box_top.top)*2):
                    min_distance = distnace
                    min_top = top
                    pair = j
        box_top_to_bottom[i] = pair

        if box_bottom_to_top.get(pair):
            box_bottom_to_top[pair].append(i)
        else:
            box_bottom_to_top[pair] = [i]



    # for key in box_bottom_to_top:
    #     if len(box_bottom_to_top[key])>1:
    #         min_distance = 9999
    #         min_value = -1
    #         for top_cell_num in box_bottom_to_top[key]:
    #             top_cell = boxes[top_cell_num]
    #             bottom_cell = boxes[key]
    #             distance = bbbox_to_distance(top_cell.bbox,bottom_cell.bbox)
    #             if min_distance>distance:
    #                 min_distance = distance
    #                 box_top_to_bottom.pop(min_value,'none')
    #                 min_value = top_cell_num
    #             else:
    #                 box_top_to_bottom.pop(top_cell_num)

    return box_top_to_bottom,box_bottom_to_top









class Layout_Analysis(object):
    def __init__(self):
        self.print_cell_list = []
        self.hand_list = []
        self.rows = []
        self.columns = []

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
            big_img = Layout_Cell([left, top, right, bottom],label = label,type='merge')
            # if '*' in big_img.label or '~' in big_img.label:
            #     big_img.type = '...'
            merge.append(big_img)

        return box_list1_,box_list2_,merge


    def row_connect(self):
        print_cell_hand = row_get_pair_by_distance(self.print_cell_list, self.hand_list)
        print_cell_residue,hand_residue,merge = self.create_big_img(print_cell_hand, self.print_cell_list, self.hand_list)
        merge_print_cell = row_get_pair_by_distance(merge,print_cell_residue)
        merge_residue,print_cell_residue,merge = self.create_big_img(merge_print_cell,merge,print_cell_residue)
        self.row_pairs = merge_residue+merge
        self.hand_after_row_connect = hand_residue
        self.print_after_row_connect = print_cell_residue
        return self.row_pairs


    def column_connect(self):
        self.all_after_row_connect = self.row_pairs+self.hand_after_row_connect+self.print_after_row_connect
        self.column_pairs,_ = column_get_pair_by_distance(self.all_after_row_connect)
        return self.column_pairs


    def intersect(self):
        for column_pair in self.column_pairs.copy():
            top = self.hand_list[column_pair]
            bottom = self.hand_list[column_pairs[column_pair]]
            column_point1 = Point(top.centre[0],top.centre[1])
            column_point2 = Point(bottom.centre[0],bottom.centre[1])
            for row_pair in self.row_pairs.copy():
                row_point1 = Point(row_pair.left,row_pair.top)
                row_point2 = Point(row_pair.right,row_pair.top)
                row_point3 = Point(row_pair.left,row_pair.bottom)
                row_point4 = Point(row_pair.right,row_pair.bottom)
                if IsIntersec(column_point1,column_point2,row_point1,row_point2) or IsIntersec(column_point1,column_point2,row_point3,row_point4):
                    self.column_pairs.pop(column_pair)
                    break

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

        for forest in forest_cell_list:
            forest.sort(key=forest_sort)


        self.forest_list = forest_cell_list



























if __name__ == '__main__':


    xml_path = config.DATA_XML
    all_img = set_xml_data(xml_path)

    for img_result in tqdm(all_img):


        layout_analysis = Layout_Analysis()

        for print_cell in img_result.print_word:
            layout_cell = Layout_Cell(print_cell.bbox,print_cell.label,'print_cell')
            layout_analysis.print_cell_list.append(layout_cell)
        random.shuffle(layout_analysis.print_cell_list)

        for hand in img_result.hand_word:
            layout_cell = Layout_Cell(hand.bbox, hand.label, 'hand')
            layout_analysis.hand_list.append(layout_cell)
        random.shuffle(layout_analysis.hand_list)

        merge = layout_analysis.row_connect()
        column_pairs = layout_analysis.column_connect()
        layout_analysis.graph_to_forest()
        # layout_analysis.intersect()
        # column_pairs = layout_analysis.column_pairs

        img = img_result.img.copy()
        img2 = img_result.img.copy()

        thickness = int(img.shape[0]/1000)+1



        for top in column_pairs:
            # top_cell = layout_analysis.hand_list[top]
            top_cell = layout_analysis.all_after_row_connect[top]
            if column_pairs[top] == -1:
                continue
            # bottom_cell = layout_analysis.hand_list[column_pairs[top]]
            bottom_cell = layout_analysis.all_after_row_connect[column_pairs[top]]
            cv2.line(img,top_cell.centre,bottom_cell.centre,color=(0, 0, 255),thickness=thickness)
            cv2.circle(img,top_cell.centre,radius = 20,color=(0,255,0),thickness = -1)
            cv2.circle(img,bottom_cell.centre,radius = 20,color=(0,255,0),thickness=-1)

        # for bbox in img_result.print_word:
        #     draw_bbox(bbox.bbox,img2,(255,0,0))
        #
        # for bbox in img_result.hand_word:
        #     draw_bbox(bbox.bbox,img2,(0,0,255))

        for bbox in layout_analysis.all_after_row_connect:
            draw_bbox(bbox.bbox,img2,(0,0,255))

        #
        # for bbox in img_result.print_cell_cell_word:
        #     draw_bbox(bbox.bbox,img2,(255,0,0))
        #
        # for bbox in img_result.hand_word:
        #     draw_bbox(bbox.bbox,img2,(0,0,255))





        cv2.imwrite(img_result.img_path.replace('data/img','layout_analysis5'),img)
        cv2.imwrite(img_result.img_path.replace('data/img', 'layout_analysis6').replace('.','_.'), img2)


