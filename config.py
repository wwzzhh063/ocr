import os


class Config(object):

    MNIST_PATH = os.environ['HOME']+ '/mnist'

    CLEAN_DATA =  os.environ['HOME']+'/clean_data'

    NOISE_DATA =  os.environ['HOME']+'/noise_data'

    VAL_DATA = './test_data'

    ONE_HOT = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'+':10,'-':11,'=':12,'×':13,'÷':14,'(':15,')':16,'':-1}

    FONT_DATA = './Fonts'

    EQUATION_MAX_LENGTH = 3


    IMAGE_HEIGHT = 32

    TEST_DATA = '/home/wzh/test_data'

    MODEL_SAVE = './model2_bn/ctc.ckpt'

    SEQ_MAXSIZE = 75

    VAL_SIZE = 100

    RNN_UNITS = 256

    SHUFFLE = [0,640,3200,3840,4160,4480,9398]

    BATCH_SIZE = 32

    CLASS_NUM = 2682

    A_CLASS_NUM = 2686

    LEARN_RATE = 1e-4

    A_UNITS = 256

