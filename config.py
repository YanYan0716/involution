import tensorflow as tf
DTYPE = tf.float32

# dataset
RESIZE = 64  # 512
CROP = 64  # 384
TRAIN_PATH = '../input/cifar10/cifar/train.csv'  # kaggle
# TRAIN_PATH = '/content/cifar/train.csv'  # google
TEST_PATH = '.'
BATCH_SIZE = 32  # 512
SHUFFLE_BUFFER = 50000

# model
SIZE = (4, 4)  # (16, 16)
HIDDEN_SIZE = 768
MLP_DIM = 1024  # 3072
NUM_HEADS = 12
NUM_LAYERS = 8  # 12
ATTENTION_DROPOUT_RATE = 0.
DROPOUT_RATE = 0.1
CLASSIFIER = 'token'
REPRESENTATION_SIZE = None
FORWARD_EXPANSION = 4
NUM_CLASSES = 10

# train
WARMUP_STEPS = 10000
BASE_LR = 0.0003
DECAY_TYPE = 'cosine'  # 'linear'
GRAD_NORM_CLIP = 1.
TOTAL_EPOCHS = 100
TOTAL_STEPS = TOTAL_EPOCHS*int((50000/BATCH_SIZE)-1)
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