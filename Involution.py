import tensorflow as tf
from tensorflow.keras.layers import Layer


class Invo2D(Layer):
    def __init__(self):
        super(Invo2D, self).__init__()

    def call(self, inputs, **kwargs):
        return inputs