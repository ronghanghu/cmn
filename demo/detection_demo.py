from __future__ import absolute_import, division, print_function

import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import skimage.io
import colorsys

from models import fastrcnn_text_objdet_model as detmodel
from util import text_processing
from util.fastrcnn_train.prepare_batch import prepare_one_image_roi

# RPN Proposal Method
import rpn_boxes

# Model path
pretrained_model = './demo_model.tfmodel'
rpn_model_file = '../models/convert_caffemodel/tfmodel/fasterrcnn_vgg_coco_net.tfmodel'
vocab_file = '../word_embedding/vocabulary_72700.txt'

# old set of variables, before this module being imported
_existing_vars = set(tf.global_variables())

# Params
T = 20
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000
mlp_hidden_dims = 500

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, None])
im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 8])

# Outputs
scores = detmodel.text_objdet(text_seq_batch, im_batch, bbox_batch,
    spatial_batch, num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=False, mlp_dropout=False, grid_score=True, return_visfeat=False)

# variables in this module
module_vars = [v for v in tf.global_variables() if v not in _existing_vars]

# Load model
snapshot_saver = tf.train.Saver(module_vars)
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
snapshot_saver.restore(sess, pretrained_model)

vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
im_mean = detmodel.fastrcnn_vgg_net.channel_mean

# Initialize rpn_boxes
print(rpn_boxes.__file__)
rpn_boxes.init(sess, rpn_model_file)

def plot_bboxes_on_im(im, bboxes):
    bboxes = bboxes.reshape((-1, 4))
    color_mult = np.linspace(0, 2./3, len(bboxes))

    plot_im = im.copy()
    for n_bbox in range(len(bboxes)-1, -1, -1):
        x1, y1, x2, y2 = bboxes[n_bbox]
        color = 255 * np.array(colorsys.hsv_to_rgb(color_mult[n_bbox], 1., 1.))
        plot_im[y1:y2+1, x1:x1+5]   = color
        plot_im[y1:y2+1, x2-4:x2+1] = color
        plot_im[y1:y1+5, x1:x2+1]   = color
        plot_im[y2-4:y2+1, x1:x2+1] = color

    return plot_im

def run_on_image(im, im_path, query):
    proposal_bboxes = rpn_boxes.extract_proposal(sess, im_path)
    print('got %d RPN proposals' % len(proposal_bboxes))
    im_val, bbox_val, spatial_val = prepare_one_image_roi(im, im_mean, proposal_bboxes, min_size=600, max_size=1000)
    text_seq_val = np.array(text_processing.preprocess_sentence(query, vocab_dict, T)).reshape((T, -1))
    scores_val = sess.run(scores, {text_seq_batch: text_seq_val,
                                   im_batch: im_val,
                                   bbox_batch: bbox_val,
                                   spatial_batch: spatial_val})
    scores_val = np.squeeze(scores_val)
    return proposal_bboxes, scores_val

def run_demo(im_path, query, savepath, visualize_num=1, score_threshold=-30.0):
    im = skimage.io.imread(im_path)
    if im.ndim == 2:
        im = np.tile(im[..., np.newaxis], (1, 1, 3))
    bboxes, scores = run_on_image(im, im_path, query)
    top_score_before_thresh = np.max(scores)
    
    # Apply threshold to the scores
    keep_inds = scores >= score_threshold
    bboxes = bboxes[keep_inds]
    scores = scores[keep_inds]
    
    top_ranks = np.argsort(scores)[::-1][:visualize_num]
    bboxes_vis = bboxes[top_ranks]

    im_vis = plot_bboxes_on_im(im, bboxes_vis)
    skimage.io.imsave(savepath, im_vis)
    
    return top_score_before_thresh
