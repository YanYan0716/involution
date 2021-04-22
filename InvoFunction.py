import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


from selfconv import Conv2D


class SelfInvo(Layer):
    def __init__(self):
        super(SelfInvo, self).__init__()
        self.a = 0

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return inputs


def test():
    x = tf.convert_to_tensor(np.random.normal(size=(1, 32, 32, 3)), dtype=tf.float32)

    tfconv = Conv2D(filters=2, kernel_size=3)
    y = tfconv(x)

    # selfconv = SelfInvo()
    print(y.shape)


if __name__ == '__main__':
    test()
