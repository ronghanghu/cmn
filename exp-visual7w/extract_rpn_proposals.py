import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # using GPU 0

import numpy as np
import tensorflow as tf
import cv2
import tqdm

sys.path.append('./util/faster_rcnn_lib/')
from fast_rcnn.config import cfg
from fast_rcnn.test import im_proposal_tensorflow
from fast_rcnn.nms_wrapper import nms

# TensorFlow Network for faster RCNN
import tensorflow as tf

from models import fastrcnn_vgg_net, rpn_net
from util.cnn import fc_layer as fc
import util.io

model_file = './models/convert_caffemodel/tfmodel/fasterrcnn_vgg_coco_net.tfmodel'

# Input placeholders
input_batch = tf.placeholder(tf.float32, [1, None, None, 3])
iminfo_batch = tf.placeholder(tf.float32, [1, 3])

# Construct the graph
conv5 = fastrcnn_vgg_net.vgg_conv5(input_batch, 'vgg_net')
rois = rpn_net.rpn_net(conv5, iminfo_batch, 'vgg_net', anchor_scales=(4, 8, 16, 32), phase='TEST')

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False)))
saver = tf.train.Saver()
saver.restore(sess, model_file)

sess_tuple = (sess, input_batch, iminfo_batch, rois)
def extract_proposal(im_file):
    im = cv2.imread(im_file)
    boxes = im_proposal_tensorflow(sess_tuple, im)
    return boxes

def extract_dataset_proposal(image_data_file, image_dir):
    image_info = util.io.load_json(image_data_file)['images']
    proposal_dict = {}
    num_img = len(image_info)
    for n_img in tqdm.trange(num_img):
        im_name = image_info[n_img]['filename']
        proposal_dict[im_name] = extract_proposal(os.path.join(image_dir, im_name))
    return proposal_dict

proposals_all = extract_dataset_proposal('./exp-visual7w/visual7w-dataset/datasets/visual7w-pointing/dataset.json',
                                         './exp-visual7w/visual7w-dataset/images/')
np.save('./exp-visual7w/data/visual7w_proposals_all.npy', np.array(proposals_all))
