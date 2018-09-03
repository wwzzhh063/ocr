import random
name = ['学姐','凡','花囤囤','查','豪哥','lucy','黑笔','叶子',
        '我','豪哥媳妇','学长','之哥','梁逼','昊哥','孟']
num = [i for i in range(1,16)]
random.shuffle(name)
result = dict(zip(name,num))
print(result)