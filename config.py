import os


class Config(object):

    MNIST_PATH = os.environ['HOME']+ '/mnist'

    CLEAN_DATA =  os.environ['HOME']+'/clean_data'

    NOISE_DATA =  os.environ['HOME']+'/noise_data'

    VAL_DATA = './test_data'

    NUM = '0123456789+-×÷()='

    ONE_HOT = {'<GO>': 0, '<EOS>': 1, '<UNK>': 2, '<PAD>': 3}

    for i in range(len(NUM)):
        ONE_HOT[NUM[i]] = i + 4

    ONE_HOT_SIZE = len(ONE_HOT)

    FONT_DATA = './Fonts'

    IMG_MAXSIZE = 250


    IMAGE_HEIGHT = 32

    TEST_DATA = '/home/wzh/test_data'

    MODEL_SAVE = './model_attention/ctc.ckpt'

    SEQ_MAXSIZE = 31

    VAL_SIZE = 100

    RNN_UNITS = 256

    SHUFFLE = [0,640,3200,3840,4160,4480,9398]

    BATCH_SIZE = 32

    CLASS_NUM = 2682

    A_CLASS_NUM = 2686

    LEARN_RATE = 1e-4

    A_UNITS = 256


print(Config.ONE_HOT)

