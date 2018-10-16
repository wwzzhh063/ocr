import tensorflow as tf
from keras.layers import Input
import cv2
import Levenshtein
import matplotlib.pyplot as plt
from pylab import *
import cutdata

img = cv2.imread('IMG_5072.JPG')
img1 = cutdata.preprocess(img)
img2 = cutdata.preprocess(img,size = 3)
cv2.imwrite('enhance1.jpg',img1)
cv2.imwrite('enhance2.jpg',img2)



