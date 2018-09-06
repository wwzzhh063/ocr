import os
from glob import glob

test_data = glob('/home/wzh/test_data/*')
for i in test_data:
    path = i.replace('test_data','test_data2')
    path = path.replace('.','_.')
    os.system('cp {} {}'.format(i,path))