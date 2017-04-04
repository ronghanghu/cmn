from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# components
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util.roi_pooling import roi_pool

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def vgg_conv5(input_batch, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        # layer 1
        conv1_1 = conv_relu('conv1_1', input_batch,
                            kernel_size=3, stride=1, output_dim=64)
        conv1_2 = conv_relu('conv1_2', conv1_1,
                            kernel_size=3, stride=1, output_dim=64)
        pool1 = pool('pool1', conv1_2, kernel_size=2, stride=2)
        # layer 2
        conv2_1 = conv_relu('conv2_1', pool1,
                            kernel_size=3, stride=1, output_dim=128)
        conv2_2 = conv_relu('conv2_2', conv2_1,
                            kernel_size=3, stride=1, output_dim=128)
        pool2 = pool('pool2', conv2_2, kernel_size=2, stride=2)
        # layer 3
        conv3_1 = conv_relu('conv3_1', pool2,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_2 = conv_relu('conv3_2', conv3_1,
                            kernel_size=3, stride=1, output_dim=256)
        conv3_3 = conv_relu('conv3_3', conv3_2,
                            kernel_size=3, stride=1, output_dim=256)
        pool3 = pool('pool3', conv3_3, kernel_size=2, stride=2)
        # layer 4
        conv4_1 = conv_relu('conv4_1', pool3,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_2 = conv_relu('conv4_2', conv4_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv4_3 = conv_relu('conv4_3', conv4_2,
                            kernel_size=3, stride=1, output_dim=512)
        pool4 = pool('pool4', conv4_3, kernel_size=2, stride=2)
        # layer 5
        conv5_1 = conv_relu('conv5_1', pool4,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_2 = conv_relu('conv5_2', conv5_1,
                            kernel_size=3, stride=1, output_dim=512)
        conv5_3 = conv_relu('conv5_3', conv5_2,
                            kernel_size=3, stride=1, output_dim=512)
    return conv5_3

def vgg_roi_fc7_from_conv5(conv5, roi_batch, name, apply_dropout, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        # ROI Pooling
        roi_pool5, _ = roi_pool(conv5, roi_batch, pooled_height=7, pooled_width=7,
            spatial_scale=1./16, name='roi_pool5')
        # layer 6
        fc6 = fc_relu('fc6', roi_pool5, output_dim=4096)
        if apply_dropout: fc6 = drop(fc6, 0.5)
        # layer 7
        fc7 = fc_relu('fc7', fc6, output_dim=4096)
        if apply_dropout: fc7 = drop(fc7, 0.5)
    return fc7

def vgg_roi_fc7(input_batch, roi_batch, name, apply_dropout, reuse=None):
    conv5 = vgg_conv5(input_batch, name, reuse=reuse)
    fc7 = vgg_roi_fc7_from_conv5(conv5, roi_batch, name, apply_dropout, reuse=reuse)
    return fc7
