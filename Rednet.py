import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import models


from Involution import Invo2D


class Bottleneck(layers.Layer):
    def __init__(self, out_channels, expansion, stride, downsample=None):
        """
        resnet中的每一个小块，对于resnet26，有26个此部分
        :param out_channels: 多少个核
        :param expansion:
        :param stride:
        :param downsample:
        """
        super(Bottleneck, self).__init__()
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0, 'in Rednet.Bottleneck, something is wrong'
        self.mid_channels = out_channels / expansion
        self.stride = stride
        self.conv1_stride = 1
        self.conv2_stride = stride
        self.downsample = downsample

        self.conv1 = keras.layers.Conv2D(
            filters=self.mid_channels,
            kernel_size=1,
            strides=self.conv1_stride,
            use_bias=False,
        )
        self.norm1 = keras.layers.BatchNormalization(name='norm1')
        self.conv2 = Invo2D()
        self.norm2 = keras.layers.BatchNormalization(name='norm2')
        self.conv3 = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            use_bias=False,
        )
        self.norm3 = keras.layers.BatchNormalization(name='norm3')
        self.relu = keras.activations.relu()


    def call(self, inputs, **kwargs):
        identity = inputs

        out = self.conv1(inputs)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        out += identity
        out = self.relu(out)
        return out


def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError(f'expansion numst be an integer or None')
    return expansion


class ResLayer(layers.Layer):
    def __init__(
            self,
            block,
            num_blocks,
            in_channels,
            out_channels,
            expansion=None,
            stride=1,
            avg_down=False,

    ):
        super(ResLayer, self).__init__()
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    keras.layers.AvgPool2D(pool_size=stride, strides=stride, padding=False)
                )
            downsample.extend([
                keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=conv_stride, use_bias=False)
                keras.layers.BatchNormalization()
            ])


    def call(self, inputs, **kwargs):
        pass


class RedNet(models.Model):
    def __init__(
            self,
            depth,
            stem_channels=64,
            base_channels=64,
            expansion=None,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(3, ),
            avg_down=False,
            frozen_stages=-1,
            zero_init_residual=True,
    ):
        super(RedNet, self).__init__()
        self.arch_settings = {
            26: (Bottleneck, (1, 2, 4, 1)),
            38: (Bottleneck, (2, 3, 5, 2)),
            50: (Bottleneck, (3, 4, 6, 3)),
            101: (Bottleneck, (3, 4, 23, 3)),
            152: (Bottleneck, (3, 8, 36, 3))
        }

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[: num_stages]
        self.strides = strides
        self.dilations = dilations
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            res_layer = ResLayer()
