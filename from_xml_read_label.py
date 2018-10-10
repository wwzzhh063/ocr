from glob import glob
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import subprocess
import random

num = 0
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


def create_dataet():
    num = 0
    for i in tqdm(glob('/home/wzh/suanshi_train/*')):

        for xml in glob(os.path.join(i, 'outputs/*')):
            other = [path for path in glob(os.path.join(i, '*')) if 'outputs' not in path][0]

            label, name = read_label(xml)

            path = os.path.join(other, name)

            if label != 'None' and label != 'Good' and label != '':
                subprocess.call(['cp', path, '/home/wzh/ocr_train/' + str(num) + '_' + label + '.jpg'])

                num = num + 1

    # onthot = open('./onehot.txt','w')
    # onthot.write(str(list))
    # print(list)

# create_dataet()



#查看xml是否有问题
# except:
#     print(xml)
# try:
#     label = root.find('outputs/transcript').text
#     name = root.find('path').text
# except:
# print(xml)
# pass
# if not label:
#     print(xml)
# print(label)




