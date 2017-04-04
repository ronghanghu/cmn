from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.rpn import ProposalLayer

def rpn_net(conv5, im_info, name, feat_stride=16, anchor_scales=(8, 16, 32),
            phase='TEST'):
    with tf.variable_scope(name):
        # rpn_conv/3x3
        rpn_conv = conv_relu('rpn_conv/3x3', conv5, kernel_size=3, stride=1,
                             output_dim=512)
        # rpn_cls_score
        # Note that we've already subtracted the bg weights from fg weights
        # and do sigmoid instead of softmax (actually sigmoid is not needed
        # for ranking)
        rpn_cls_score = conv('rpn_cls_score', rpn_conv, kernel_size=1, stride=1,
                             output_dim=len(anchor_scales)*3)
        # rpn_bbox_pred
        rpn_bbox_pred = conv('rpn_bbox_pred', rpn_conv, kernel_size=1, stride=1,
                             output_dim=len(anchor_scales)*3*4)

        rois = tf.py_func(ProposalLayer(feat_stride, anchor_scales, phase),
                          [rpn_cls_score, rpn_bbox_pred, im_info],
                          [tf.float32], stateful=False)[0]
        rois.set_shape([None, 5])
        return rois
