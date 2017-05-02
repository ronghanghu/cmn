from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import deconv_layer as deconv
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from models import fastrcnn_vgg_net, vgg_net, lstm_net
from models.processing_tools import *

def _grid_add(tensor1, tensor2, D):
    """
    broadcast and add two tensors
       tensor1 is [N1, D]
       tensor2 is [N2, D]
       output is [N1, N2, D]
    """
    tensor1 = tf.reshape(tensor1, [-1, 1, D])
    tensor2 = tf.reshape(tensor2, [1, -1, D])
    return tf.add(tensor1, tensor2)

def text_objdet(text_seq_batch, im_batch, bbox_batch, spatial_batch,
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims, vgg_dropout, mlp_dropout,
    grid_score, return_visfeat=True):
    """
    If grid_score is true, the output will be [N_lan x N_vis, 1] format
    and can be reshaped to [N_lan, N_vis, 1]

    Otherwise it assumes N_vis == N_lan or N_vis == 1 or N_lan == 1
    """

    # Local image feature
    feat_fc7 = fastrcnn_vgg_net.vgg_roi_fc7(im_batch, bbox_batch, 'vgg_local',
        apply_dropout=vgg_dropout)
    # All local features: image feature and spatial feature of the bounding box
    # No L2-normalization is be used (it does not seem necessary)
    feat_vis = tf.concat([feat_fc7, spatial_batch], axis=1)

    # Language feature (LSTM hidden state)
    feat_lan = lstm_net.lstm_encoder(text_seq_batch, 'lstm', num_vocab,
                                     embed_dim, lstm_dim, apply_dropout=False)

    # A first layer of classifier thats process image features and text features
    # separately for later broadcast-able concatenation
    with tf.variable_scope('classifier'):
        # First layer in the classifier
        mlp_l1_vis = fc('mlp_l1_vis', feat_vis, output_dim=mlp_hidden_dims)
        # one bias term is sufficient
        mlp_l1_lan = fc('mlp_l1_lan', feat_lan, output_dim=mlp_hidden_dims,
                             bias_term=False)
        # Add up mlp_l1_vis and mlp_l1_lan
        if grid_score:
            # Put query number at the first dimension
            mlp_l1 = _grid_add(mlp_l1_lan, mlp_l1_vis, mlp_hidden_dims)
            mlp_l1 = tf.reshape(mlp_l1, [-1, mlp_hidden_dims])
        else:
            mlp_l1 = tf.add(mlp_l1_vis, mlp_l1_lan)
        # Apply nonlinearity
        mlp_l1 = tf.nn.relu(mlp_l1)
        if mlp_dropout:
            mlp_l1 = drop(mlp_l1, 0.5)

        # Second layer in the classifier
        mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)
    if return_visfeat:
        return mlp_l2, feat_vis  # feat_vis also includes spatial features

    return mlp_l2

def text_imgcls(text_seq_batch, im_batch, spatial_batch, num_vocab, embed_dim,
    lstm_dim, mlp_hidden_dims, vgg_dropout, mlp_dropout, grid_score):
    """
    If grid_score is true, the output will be [N_lan x N_vis, 1] format
    and can be reshaped to [N_lan, N_vis, 1]

    Otherwise it assumes N_vis == N_lan or N_vis == 1 or N_lan == 1
    """

    # Local image feature
    feat_fc7 = vgg_net.vgg_fc7(im_batch, 'vgg_local',
        apply_dropout=vgg_dropout)
    # All local features: image feature and spatial feature of the bounding box
    # No L2-normalization is be used (it does not seem necessary)
    feat_vis = tf.concat(1, [feat_fc7, spatial_batch])

    # Language feature (LSTM hidden state)
    feat_lan = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)

    # A first layer of classifier thats process image features and text features
    # separately for later broadcast-able concatenation
    with tf.variable_scope('classifier'):
        # First layer in the classifier
        mlp_l1_vis = fc('mlp_l1_vis', feat_vis, output_dim=mlp_hidden_dims)
        # one bias term is sufficient
        mlp_l1_lan = fc('mlp_l1_lan', feat_lan, output_dim=mlp_hidden_dims,
                             bias_term=False)
        # Add up mlp_l1_vis and mlp_l1_lan
        if grid_score:
            # Put query number at the first dimension
            mlp_l1 = _grid_add(mlp_l1_lan, mlp_l1_vis, mlp_hidden_dims)
            mlp_l1 = tf.reshape(mlp_l1, [-1, mlp_hidden_dims])
        else:
            mlp_l1 = tf.add(mlp_l1_vis, mlp_l1_lan)
        # Apply nonlinearity
        mlp_l1 = tf.nn.relu(mlp_l1)
        if mlp_dropout:
            mlp_l1 = drop(mlp_l1, 0.5)

        # Second layer in the classifier
        mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)

    return mlp_l2
