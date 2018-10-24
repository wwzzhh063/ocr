
import cv2

import matplotlib.pyplot as plot


from config import Config as config
from glob import glob
from tqdm import tqdm
import os

path1 = os.path.join(config.VAL_DATA,'*')
path2 = os.path.join(config.CLEAN_DATA,'*')


img_len_wide = []                  #图片长宽比
label_len_list = []
char_len_resize_list = []
for img_path in tqdm(glob(path1)+glob(path2)):
    img = cv2.imread(img_path)
    proportion = img.shape[1] / img.shape[0]
    label = img_path.split('_')[-1].split('.')[0]
    label_len = len(label)
    img_len_wide.append(proportion)
    label_len_list.append(label_len)
    len_label = img.shape[1]/(img.shape[0]/32)/label_len
    char_len_resize_list.append(len_label)
    # if proportion>7:
    #     print(img_path)
    if len_label<5:
        print(img_path)


plot.figure(figsize=(40, 10), dpi=80)  # 绘制直方图
plot.xticks(fontsize=40)
plot.yticks(fontsize=40)
plot.hist(img_len_wide, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plot.savefig('img_len_wide', bbox_inches='tight')


plot.figure(figsize=(40, 10), dpi=80)  # 绘制直方图
plot.xticks(fontsize=40)
plot.yticks(fontsize=40)
plot.hist(label_len_list, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plot.savefig('label_len', bbox_inches='tight')


plot.figure(figsize=(40, 10), dpi=80)  # 绘制直方图
plot.xticks(fontsize=40)
plot.yticks(fontsize=40)
plot.hist(char_len_resize_list, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plot.savefig('char_len_resize', bbox_inches='tight')
plot.show