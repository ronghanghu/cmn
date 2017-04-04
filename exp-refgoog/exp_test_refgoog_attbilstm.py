from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # using GPU 0
import time
import tensorflow as tf
import numpy as np

from models import refgoog_attention_model
from util.refgoog_baseline_train.roi_data_reader import DataReader
from util import loss
from util.eval_tools import compute_accuracy
import util.io

################################################################################
# Parameters
################################################################################

# Model Params
T = 20
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000

# Data Params
imdb_file = './exp-refgoog/data/imdb/imdb_val.npy'
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = refgoog_attention_model.fastrcnn_vgg_net.channel_mean

# Snapshot Params
snapshot_file = './downloaded_models/refgoog_attbilstm_iter_150000.tfmodel'

output_file = './exp-refgoog/results/refgoog_attbilstm_iter_150000_val.txt'

################################################################################
# The model
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, None])
im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 5])

# Outputs
scores = refgoog_attention_model.refgoog_attbilstm_net(im_batch, bbox_batch,
    spatial_batch, text_seq_batch, num_vocab, embed_dim, lstm_dim,
    vgg_dropout=False, lstm_dropout=False)

################################################################################
# Initialize parameters and load data
################################################################################

# Load data
reader = DataReader(imdb_file, vocab_file, im_mean, shuffle=False)

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Snapshot saver
snapshot_saver = tf.train.Saver()
snapshot_saver.restore(sess, snapshot_file)

################################################################################
# Optimization loop
################################################################################

num_correct = 0
num_total = 0

# Run optimization
for n_iter in range(reader.num_batch):
    # Read one batch
    batch = reader.read_batch()
    text_seq_val  = batch['text_seq_batch']
    im_val        = batch['im_batch']
    bbox_val      = batch['bbox_batch']
    spatial_val   = batch['spatial_batch']

    # Forward and Backward pass
    scores_val = sess.run(scores,
        feed_dict={
            text_seq_batch  : text_seq_val,
            im_batch        : im_val,
            bbox_batch      : bbox_val,
            spatial_batch   : spatial_val
        })

    predicts = np.argmax(scores_val, axis=1)

    # save json
    if n_iter == 0:
        eval_output_json = []
    for n_sentence in range(len(predicts)):
        result = {
            "annotation_id": batch["coco_ann_ids"][n_sentence],
            "predicted_bounding_boxes": [list(batch["coco_bboxes"][predicts[n_sentence]])],
            "refexp": batch["questions"][n_sentence]
        }
        eval_output_json.append(result)
    if n_iter == 0: # check if save passes..
        util.io.save_json(eval_output_json, output_file)
    if n_iter == reader.num_batch - 1:
        util.io.save_json(eval_output_json, output_file)
        print('evaluation output file saved to %s' % output_file)
