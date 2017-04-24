import sys
import numpy as np
import tensorflow as tf
import cv2

sys.path.append('../util/faster_rcnn_lib/')
from fast_rcnn.config import cfg
from fast_rcnn.test import im_proposal_tensorflow
from fast_rcnn.nms_wrapper import nms

# TensorFlow Network for faster RCNN
from models import fastrcnn_vgg_net, rpn_net

# old set of variables, before this module being imported
_existing_vars = set(tf.global_variables())

# Input placeholders
input_batch = tf.placeholder(tf.float32, [1, None, None, 3])
iminfo_batch = tf.placeholder(tf.float32, [1, 3])

# Construct the graph
conv5 = fastrcnn_vgg_net.vgg_conv5(input_batch, 'vgg_net')
rois = rpn_net.rpn_net(conv5, iminfo_batch, 'vgg_net', anchor_scales=(4, 8, 16, 32), phase='TRAIN')

# The variables in RPN
rpn_vars = [v for v in tf.global_variables() if v not in _existing_vars]

def init(sess, model_file):
    saver = tf.train.Saver(rpn_vars)
    saver.restore(sess, model_file)

def extract_proposal(sess, im_file, max_proposal_num=2500):
    im = cv2.imread(im_file)
    sess_tuple = (sess, input_batch, iminfo_batch, rois)
    boxes = im_proposal_tensorflow(sess_tuple, im)
    boxes = boxes.astype(np.int32)
    boxes = boxes[:max_proposal_num]
    return boxes
