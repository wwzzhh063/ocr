from cutdata import ParseXml

from cutdata import ParseXml
import os
from glob import glob
import cv2
from sklearn.metrics.pairwise import pairwise_distances
import math


def get_pt_wd_hd_wd(path):
    xml_path = glob(os.path.join(path, '*.xml'))
    pt_wd = []
    hd_wd = []
    img_name = []
    for i in xml_path:
        p = ParseXml(i)
        pt_wd_ = []
        hd_wd_ = []
        img_name_, class_list, bbox_list = p.get_bbox_class()
        img_name.append(img_name_)
        for i,type in enumerate(class_list):
            if type == 1:
                pt_wd_.append(bbox_list[i])
            else:
                hd_wd_.append(bbox_list[i])
        pt_wd.append(pt_wd_)
        hd_wd.append(hd_wd_)
    return pt_wd,hd_wd,img_name

# def list_pt_wd(pt_wds):
#     list_all = []
#     for pt_wd in pt_wds:
#         if len()
        





def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))


def bbbox_to_distance(point_i,point_j):
    if len(point_i) == 4:
        pt_wd_point = (point_i[2], (point_i[1] + point_i[3]) / 2)
    else:
        pt_wd_point = ((point_i[2] + point_i[4]) / 2, (point_i[3] + point_i[5]) / 2)

    if len(point_j) == 4:
        hd_wd_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
    else:
        hd_wd_ponit = ((point_j[0] + point_j[6]) / 2, (point_j[1] + point_j[7]) / 2)

    distences = get_distance(hd_wd_ponit, pt_wd_point)


    return distences

def bbbox_to_distance2(point_i,point_j):
    if len(point_i) == 4:
        pt_wd_point = (point_i[2], (point_i[1] + point_i[3]) / 2)
    else:
        pt_wd_point = ((point_i[2] + point_i[4]) / 2, (point_i[3] + point_i[5]) / 2)

    if len(point_j) == 4:
        hd_wd_ponit = (point_j[0], (point_j[1] + point_j[3]) / 2)
    else:
        hd_wd_ponit = ((point_j[0] + point_j[6]) / 2, (point_j[1] + point_j[7]) / 2)

    distences = get_distance(hd_wd_ponit, pt_wd_point)

    return to_int(pt_wd_point),to_int(hd_wd_ponit)


def to_int(a):
    x,y = a
    return (int(x),int(y))

def get_pair(pt_wd,hd_wd):
    a = []
    pairs_pt_hd = {}
    pairs_hd_pt = {}
    for i ,point_i in enumerate(pt_wd):
        length_min = 999
        pair = 0
        for j ,point_j in enumerate(hd_wd):

            distences = bbbox_to_distance(point_i,point_j)

            if distences<length_min:
                length_min = distences
                pair = j

        pairs_pt_hd[i] = pair

        if pairs_hd_pt.get(pair):
            pairs_hd_pt[pair].append(i)
        else:
            pairs_hd_pt[pair] = [i]

    for key in pairs_hd_pt:
        value = pairs_hd_pt[key]
        if len(value)>1:
            distences = bbbox_to_distance(hd_wd[key],pt_wd[value[0]])
            j = 0
            for i in range(1, len(value)):
                if distences > bbbox_to_distance(hd_wd[key], pt_wd[value[i]]):
                    pairs_pt_hd.pop(value[j])
                    distences = bbbox_to_distance(hd_wd[key], pt_wd[value[i]])
                    j = i
                else:
                    pairs_pt_hd.pop(value[i])

    return pairs_pt_hd

def get_centre(box):
    if len(box) == 4:
        point = (int((box[2]+box[0])/2), int((box[1] + box[3]) / 2))
    else:
        point = (int((box[0] + box[4]) / 2), int((box[1] + box[5]) / 2))

    return point


def write_line(img,pt_wd,hd_wd,pairs):
    for key in pairs:
        x,y = bbbox_to_distance2(pt_wd[key],hd_wd[pairs[key]])
        cv2.line(img,  x,y, (255))
        # cv2.line(img,get_centre(pt_wd[key]),get_centre(hd_wd[pairs[key]]),(255))





def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))



if __name__ == '__main__':
    xml_path = '/home/wzh/ocr-demo/outputs/'
    pt_wd, hd_wd, img_name = get_pt_wd_hd_wd(xml_path)

    img_name = img_name[0].replace('xml', 'JPG')
    img_path = '/home/wzh/ocr-demo/pic/'
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    img = cv2.imread(img_path + img_name)


    # for i, point_i in enumerate(pt_wd[0]):
    #     for j, point_j in enumerate(hd_wd[0]):
    #         pt_wd_point, hd_wd_ponit = bbbox_to_distance(point_i, point_j)
    #         cv2.circle(img, pt_wd_point, 10, (255, 0, 0))
    #         cv2.circle(img, hd_wd_ponit, 10, (255, 0, 0))



    pair = get_pair(pt_wd[0], hd_wd[0])
    write_line(img,pt_wd[0], hd_wd[0],pair)
    cv2.imwrite("1.jpg",img)
    cv2.waitKey()