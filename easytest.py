import tensorflow as tf
from keras.layers import Input
import cv2
import Levenshtein

a = '32+5=37'
b = '32+5=31'

print(Levenshtein.distance(a,b))


