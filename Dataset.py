import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd


import config


def train_image(img_file, label):
    # 对图片的处理
    img = tf.io.read_file(os.path.join(config.ROOT_PATH, img_file))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, (config.RESIZE, config.RESIZE))
    img = tf.image.random_crop(img, (config.CROP, config.CROP, 3))
    img = tf.cast(img, config.DTYPE) / 255.0
    mean = tf.expand_dims(tf.convert_to_tensor([0.4914, 0.4822, 0.4465], dtype=config.DTYPE), axis=0)
    std = tf.expand_dims(tf.convert_to_tensor([0.2471, 0.2435, 0.2616], dtype=config.DTYPE), axis=0)
    img = (img-mean)/std

    label = tf.cast(label, dtype=config.DTYPE)
    return img, label


def test_image(img_file, label):
    # 对图片的处理
    img = tf.io.read_file(os.path.join(config.ROOT_PATH, img_file))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.CROP, config.CROP))
    img = tf.cast(img, config.DTYPE) / 255.0
    mean = tf.expand_dims(tf.convert_to_tensor([0.4914, 0.4822, 0.4465], dtype=config.DTYPE), axis=0)
    std = tf.expand_dims(tf.convert_to_tensor([0.2471, 0.2435, 0.2616], dtype=config.DTYPE), axis=0)
    img = (img-mean)/std

    label = tf.cast(label, dtype=config.DTYPE)
    return img, label


if __name__ == '__main__':

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 有标签的数据集 batch_size=config.BATCH_SIZE
    df_label = pd.read_csv(config.TRAIN_PATH)
    file_paths = df_label['file_name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train\
        .map(label_image, num_parallel_calls=AUTOTUNE)\
        .batch(1)\
        .shuffle(1)
    for data in ds_label_train:
        print(data.keys())
        break