import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Invo2D(Layer):
    def __init__(self, filters, kernel_size, stride):
        super(Invo2D, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        reduction_ratio = 4
        self.group_channels = 16
        self.group = self.filters // self.group_channels

        self.conv1 = keras.layers.Conv2D(
            filters=filters // reduction_ratio,
            kernel_size=1,
        )
        self.norm1 = keras.layers.BatchNormalization()
        self.relu1 = keras.activations.relu

        self.conv2 = keras.layers.Conv2D(
            filters=self.kernel_size ** 2 * self.group,
            kernel_size=1,
            strides=1,
        )
        if self.stride > 1:
            self.avgpool = keras.layers.AvgPool2D(strides=(stride, stride))

    def call(self, inputs, **kwargs):
        weights = self.conv2(self.conv1(inputs if self.stride == 1 else self.avgpool(inputs)))
        b, h, w, c = weights.shape
        weights = tf.transpose(weights, perm=[0, 3, 1, 2])
        weights = tf.reshape(
            tensor=weights,
            shape=(b, self.group, self.kernel_size ** 2, h, w)
        )
        weights = tf.expand_dims(weights, axis=2)
        out = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        out = tf.transpose(out, perm=[0, 3, 1, 2])
        out = tf.reshape(out, shape=(b, self.group, self.group_channels, self.kernel_size ** 2, h, w))
        out = tf.math.multiply(weights, out)
        out = tf.reshape(tf.reduce_sum(out, axis=3), shape=(b, self.filters, h, w))
        out = tf.transpose(out, perm=[0, 2, 3, 1])
        return out


def test():
    """how to use tf.image.extract_patches, it is reaponse to torch.nn.unfold"""
    import numpy as np
    # img = tf.convert_to_tensor(np.random.random(size=(2, 3, 4, 5)))
    # y = tf.image.extract_patches(
    #     images=img,
    #     sizes=[1, 2, 3, 1],
    #     strides=[1, 1, 1, 1],
    #     rates=[1, 1, 1, 1],
    #     padding='VALID'
    # )
    img = tf.convert_to_tensor(np.random.random(size=(2, 16, 16, 128)), dtype=tf.float32)
    Involayer = Invo2D(filters=128, kernel_size=7, stride=1)
    y = Involayer(img)

    print(y.shape)


if __name__ == '__main__':
    test()
