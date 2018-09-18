from glob import glob
import os
import config

dict = {}
for i ,j in enumerate(config.Config.ONE_HOT):
    dict[j] = i
dict[''] = -1
print(dict)