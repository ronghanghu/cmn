import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--begin_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=-1)
parser.add_argument('--gpu_id', default='0')
args = parser.parse_args()
import os; os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import numpy as np
import tensorflow as tf
import cv2
import tqdm

import sys; sys.path.append('./util/faster_rcnn_lib/')
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
rois = rpn_net.rpn_net(conv5, iminfo_batch, 'vgg_net', anchor_scales=(4, 8, 16, 32), phase='TRAIN')

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False)))
saver = tf.train.Saver()
saver.restore(sess, model_file)

sess_tuple = (sess, input_batch, iminfo_batch, rois)
def extract_proposal(im_file):
    im = cv2.imread(im_file)
    boxes = im_proposal_tensorflow(sess_tuple, im)
    return boxes

def extract_dataset_proposal(image_list_file, image_dir, save_dir, begin_idx=0, end_idx=-1):
    with open(image_list_file) as f:
        image_names = [l.split()[1] for l in f.readlines()]
    image_names = image_names[begin_idx:end_idx]

    num_img = len(image_names)
    for n_img in tqdm.trange(num_img):
        im_name = image_names[n_img]
        image_path = os.path.join(image_dir, im_name)
        save_path = os.path.join(save_dir, im_name.replace('.jpg', '.npy'))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        bboxes = extract_proposal(image_path).astype(np.int16)
        np.save(save_path, bboxes)

extract_dataset_proposal('./exp-cub200/cub200-dataset/images.txt',
                         './exp-cub200/cub200-dataset/images/',
                         './exp-cub200/data/rpn_proposals/',
                         args.begin_idx, args.end_idx)
