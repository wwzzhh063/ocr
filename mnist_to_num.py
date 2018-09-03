import numpy as np
import cv2
import os
from config import Config
from glob import glob
import random
from PIL import Image,ImageDraw,ImageFont
from tqdm import *
import skimage
from captcha.image import ImageCaptcha

path = sorted(glob(os.path.join(Config.MNIST_PATH,'*')))

image_path = []

for i in range(10):
    image_path.append(glob(os.path.join(path[i],'*')))


signs = ['+','-','×','÷','(',')']

size_list = [25,22,19,16]
step_size_dict = {25:14,22:13,19:11,16:10}
head_size_dict = {25:2,22:4,19:6,16:8}
color_list = [0,50,100,120]
noise_list = ['gaussian','salt']






def create_data(length,ttfont,head_size,color,background_path):
    label = []
    head_empty = random.randint(10,20)
    end_empty = random.randint(10,20)
    step_size = ttfont.getsize('txt')[0]*0.5

    for i in range(length):
        num  = random.randint(0,1)
        if num == 0:
            num = random.randint(0,9)
        else:
            num = random.randint(10,99)
        num_list = list(str(num))
        if i == length-1:
            sign = ['=']
        else:
            sign = random.sample(signs,1)
        label.extend(num_list)
        label.extend(sign)
    # image = np.zeros([28, 60 + 18 *len(label)])

    image = Image.new('L',(int(56+head_empty+end_empty + step_size *len(label)),32), 255)
    draw = ImageDraw.Draw(image)
    draw.text((head_empty+head_size, head_size), ''.join(label), fill='black', font=ttfont)
    image = np.asarray(image)
    image.flags.writeable = True




    num = random.randint(0, 99)
    num_list = list(str(num))
    for i, num in enumerate(num_list):
        path = random.sample(image_path[int(num)], 1)[0]
        image_num = cv2.imread(path)
        image_num = cv2.cvtColor(image_num, cv2.COLOR_BGR2GRAY)
        image_num = 255-image_num
        image[2:30, image.shape[1]-56-end_empty+28 * i:image.shape[1]-56-end_empty+28 * (i + 1)] = image_num
        label.append(num)

    if background_path=='random':
        rgb = random.randint(170, 255)
        background = np.zeros([image.shape[0],image.shape[1],3])+rgb
    else:
        background = cv2.imread(background_path)
        background = cv2.resize(background,(image.shape[1],image.shape[0]))
    background_mask = np.floor(image/240)[...,np.newaxis]
    background = background * background_mask
    color_mask = 1-background_mask
    background = background+color_mask*color

    # image_mask = 1-background_mask
    # image = image.astype(np.uint8)
    # image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    # image = image*image_mask
    # cv2.imwrite('d.jpg',background)
    # #
    # image = image+background

    generator = ImageCaptcha(width=400, height=80)

    if random.randint(0,2) == 1:
        background = generator.generate_image(''.join(label))
        background = np.asarray(background, np.uint8)


    return background,label

def create_dataset(noise=True):
    if noise:
        path = Config.NOISE_DATA
    else:
        path = Config.CLEAN_DATA
        background_path = './background.png'
    if not os.path.exists(path):
        os.mkdir(path)

    font_list = glob(os.path.join(Config.FONT_DATA,'*.ttf'))
    font_list.extend(glob(os.path.join(Config.FONT_DATA,'*.TTF')))

    for i in tqdm(range(500)):
        if noise:
            background_path = random.sample(['./background.png', './background_noise.png', 'random'], 1)[0]
        size = random.sample(size_list,1)[0]
        head_size = head_size_dict[size]
        color = random.sample(color_list,1)[0]
        noise_type = random.sample(noise_list, 1)[0]
        ttfont = ImageFont.truetype(random.sample(font_list,1)[0],size)
        length = random.randint(2,3)
        image,label = create_data(length,ttfont,head_size,color,background_path)
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        temp = 'clean'
        if noise:
                image = image.astype(np.uint8)
                image = skimage.util.random_noise(image, mode=noise_type, seed=None, clip=True)*255
                temp = 'noise'
                image = image.astype(np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path,str(i)+temp+'_'+''.join(label)+'.jpg'),image)

create_dataset()

