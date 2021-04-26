import tensorflow as tf
DTYPE = tf.float32

# dataset
RESIZE = 256  # 512
CROP = 224  # 384
TRAIN_PATH = '../input/cifar10/cifar/train.csv'  # kaggle
# TRAIN_PATH = '/content/cifar/train.csv'  # google
BATCH_SIZE = 32  # 512
SHUFFLE_BUFFER = 50000

# model
NUM_CLASSES = 10

# train
BASE_LR = 0.0003
TOTAL_EPOCHS = 100
CONTINUE = True
START_EPOCH = 0


# test
TEST_PATH = '../input/cifar10/cifar/test.csv'  # kaggle
# TEST_PATH = '/content/cifar/test.csv'  # google
LOAD_PATH = '../input/weights/weights/M'

# evaluate
LOG_EPOCH = 2
LOG_LOSS = 200
SAVE_PATH = './weights/M'