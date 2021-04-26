import tensorflow as tf
DTYPE = tf.float32

# dataset
RESIZE = 36  # 512
CROP = 32  # 384
ROOT_PATH = ''
# TRAIN_PATH = '../input/cifar10/cifar/train.csv'  # kaggle
TRAIN_PATH = '/content/cifar/train.csv'  # google
BATCH_SIZE = 128  # 512
SHUFFLE_BUFFER = 50000

# model
NUM_CLASSES = 10

# train
BASE_LR = 0.0003
TOTAL_EPOCHS = 100
CONTINUE = False
START_EPOCH = 0


# test
# TEST_PATH = '../input/cifar10/cifar/test.csv'  # kaggle
TEST_PATH = '/content/cifar/test.csv'  # google
LOAD_PATH = '/content/drive/MyDrive/weights/Invo'

# evaluate
LOG_EPOCH = 2
LOG_LOSS = 100
SAVE_PATH = '/content/drive/MyDrive/weights/Invo'