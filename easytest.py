import tensorflow as tf
from keras.layers import Input
import cv2
import Levenshtein
import matplotlib.pyplot as plot

import cutdata
import subprocess


from config import Config as config
from glob import glob
from tqdm import tqdm
import os

one_hot = config.ONE_HOT
a = set(one_hot)
print(a)
path1 = os.path.join(config.VAL_DATA,'*')
path2 = os.path.join(config.CLEAN_DATA,'*')

not_in = []
for path in tqdm(glob(path1)+glob(path2)):
    label = path.split('_')[-1].split('.')[0]
    if label=='':
        subprocess.call(['rm', path])

