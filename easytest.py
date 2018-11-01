import tensorflow as tf
from keras.layers import Input
import cv2
import Levenshtein
import matplotlib.pyplot as plot

import subprocess


from config import Config as config
from glob import glob
from tqdm import tqdm
import os

a =  {'1': 0, '0': 1, '8': 2, '5': 3, '3': 4, '-': 5, '2': 6, '=': 7, '6': 8, '×': 9, '7': 10, '÷': 11, '4': 12, '9': 13, '+': 14, '捡': 15, '起': 16, '来': 17, '吧': 18, '错': 19, '题': 20, '本': 21, '把': 22, '掉': 23, '落': 24, '的': 25, '(': 26, ')': 27, '*': 28, '口': 29, '算': 30, '练': 31, '习': 32, '>': 33, '笔': 34, '闯': 35, '关': 36, '家': 37, '长': 38, '评': 39, '分': 40, ':': 41, '用': 42, '时': 43, '!': 44, '第': 45, '天': 46, '月': 47, '日': 48, '乘': 49, '号': 50, '竖': 51, '式': 52, '脱': 53, '计': 54, '获': 55, '得': 56, '收': 57, '了': 58, '巧': 59, '个': 60, '称': 61, '冲': 62, '刺': 63, '回': 64, '准': 65, '确': 66, '率': 67, '开': 68, '始': 69, '基': 70, '础': 71, '过': 72, '小': 73, '朋': 74, '友': 75, ',': 76, '打': 77, '上': 78, '对': 79, '勾': 80, '文': 81, '具': 82, '盒': 83, '才': 84, '可': 85, '以': 86, '购': 87, '物': 88, '车': 89, '里': 90, '哦': 91, '你': 92, '几': 93, '呢': 94, '?': 95, '被': 96, '能': 97, '力': 98, '提': 99, '高': 100, '@': 101, '初': 102, '级': 103, '银': 104, '员': 105, '中': 106, '钟': 107, '~': 108, '.': 109, "'": 110, '未': 111, '借': 112, '位': 113, '有': 114, '棉': 115, '花': 116, '背': 117, '篓': 118, '优': 119, '括': 120, '误': 121, '八': 122, '火': 123, '锅': 124, '菜': 125, '%': 126, '金': 127, '针': 128, '菇': 129, '好': 130, '秒': 131, '父': 132, '运': 133, '符': 134, '变': 135, '化': 136, '前': 137, '是': 138, '减': 139, '正': 140, '棒': 141, '哒': 142, '学': 143, '年': 144, '数': 145, '。': 146, '极': 147, '－': 148, '如': 149, '拣': 150, 'd': 151, '巴': 152, 'o': 153, 'G': 154, '‘': 155, 'e': 156, 'N': 157, '￥': 158, '周': 159, 'n': 160, 's': 161, 'S': 162, '加': 163,'': -1}
b = ''.join(list(set(a)))
print(b)