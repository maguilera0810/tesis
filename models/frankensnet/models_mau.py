
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from keras.applications.densenet import DenseNet201
from keras import layers
from keras import backend
from keras import models
from keras.regularizers import l2
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import imagenet_utils

# -------------------------------DENSENET--------------------------------
HEIGHT, WIDTH = 224, 224


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

# -----------------------------INCEPTIONV3------------------------------


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x


def InceptionModel_B(x, i):
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed_b_' + str(9 + i))
    return x


def InceptionModel_C(x, i):
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate(
        [branch3x3_1, branch3x3_2],
        axis=3,
        name='mixed9_' + str(i))

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = layers.concatenate(
        [branch3x3dbl_1, branch3x3dbl_2], axis=3)

    branch_pool = layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed_c_' + str(9 + i))
    return x


def FrankensNet(input_shape=None, classes=2):

    dense_model = DenseNet201(
        include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in dense_model.layers:
        layer.trainable = False
    x = dense_model.output

    branch_1 = InceptionModel_B(x, 1)
    branch_1 = InceptionModel_B(branch_1, 2)
    branch_1 = InceptionModel_B(branch_1, 3)
    branch_1 = InceptionModel_B(branch_1, 4)
    branch_1 = layers.MaxPooling2D(name='max_pool_1')(branch_1)
    branch_1 = InceptionModel_C(branch_1, 1)
    branch_1 = InceptionModel_C(branch_1, 2)

    branch_2 = InceptionModel_B(x, 5)
    branch_2 = InceptionModel_B(branch_2, 6)
    branch_2 = InceptionModel_B(branch_2, 7)
    branch_2 = layers.MaxPooling2D(name='max_pool_2')(branch_2)
    branch_2 = InceptionModel_C(branch_2, 3)
    branch_2 = InceptionModel_C(branch_2, 4)
    branch_2 = InceptionModel_C(branch_2, 5)

    x1 = layers.Conv2D(256,3)(x)
    x1 = layers.Conv2D(256,3)(x1)

    branch_3 = InceptionModel_B(x, 8)
    branch_3 = InceptionModel_B(branch_3, 9)
    branch_3 = layers.MaxPooling2D(name='max_pool_3')(branch_3)
    branch_3 = InceptionModel_C(branch_3, 6)
    branch_3 = InceptionModel_C(branch_3, 7)
    branch_3 = InceptionModel_C(branch_3, 8)
    branch_3 = InceptionModel_C(branch_3, 9)

    branches1 = layers.Concatenate()([branch_1, branch_2, branch_3,x1])

 

    x3 = layers.MaxPooling2D(name='max_pool_6')(branches1)
    x2 = layers.AveragePooling2D(name='avrg_pool_final')(branches1)
    x = layers.Concatenate()([x2, x3])
    x = layers.Flatten()(x)
    

    x = layers.Dense(1024, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.4)(x)

    predictions = layers.Dense(2, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=dense_model.input, outputs=[
                         predictions], name='frankensnet')

    return model
