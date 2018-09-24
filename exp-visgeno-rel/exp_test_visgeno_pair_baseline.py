from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # using GPU 0

import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform

from models import visgeno_baseline_model, spatial_feat, fastrcnn_vgg_net
from util.visgeno_rel_train.rel_data_reader import DataReader
from util import loss, eval_tools, text_processing

################################################################################
# Parameters
################################################################################

# Model Params
T = 20
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000

# Data Params
imdb_file = './exp-visgeno-rel/data/imdb/imdb_tst.npy'
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = visgeno_baseline_model.fastrcnn_vgg_net.channel_mean

# Snapshot Params
model_file1 = './downloaded_models/visgeno_baseline_iter_360000.tfmodel'
model_file2 = './downloaded_models/visgeno_baseline_obj2_iter_360000.tfmodel'

result_file = './exp-visgeno-rel/results/visgeno_baseline_iter_360000_visgeno_baseline_obj2_iter_360000_tst_pair.txt'

################################################################################
# Network
################################################################################

im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, None])

with tf.variable_scope("obj1"):
    scores_obj1 = visgeno_baseline_model.visgeno_baseline_net(im_batch, bbox_batch, spatial_batch,
        text_seq_batch, num_vocab, embed_dim, lstm_dim, False, False)
with tf.variable_scope("obj2"):
    scores_obj2 = visgeno_baseline_model.visgeno_baseline_net(im_batch, bbox_batch, spatial_batch,
        text_seq_batch, num_vocab, embed_dim, lstm_dim, False, False)

scores = scores_obj1[:, :, tf.newaxis, :] + scores_obj2[:, tf.newaxis, :, :]


var_obj1_list = [v for v in tf.global_variables() if v.op.name.startswith("obj1")]
var_obj2_list = [v for v in tf.global_variables() if v.op.name.startswith("obj2")]


np.random.seed(3)
reader = DataReader(imdb_file, vocab_file, im_mean, shuffle=False, max_bbox_num=10000, max_rel_num=10000)

################################################################################
# Snapshot and log
################################################################################

# Snapshot saver
var_obj1_dict = {v.op.name.replace("obj1/", ""): v for v in var_obj1_list}
var_obj2_dict = {v.op.name.replace("obj2/", ""): v for v in var_obj2_list}
snapshot_saver1 = tf.train.Saver(var_obj1_dict)
snapshot_saver2 = tf.train.Saver(var_obj2_dict)

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Run Initialization operations
snapshot_saver1.restore(sess, model_file1)
snapshot_saver2.restore(sess, model_file2)

################################################################################
# Test loop
################################################################################

K = 10
top_x_correct_count = np.zeros(K)  # compute up to top-K accuracy
total = 0

# Run test
for n_iter in range(reader.num_batch):
    batch = reader.read_batch()

    ###
    # continue if the batch does not contain any relationship
    # and increment n_iter so that the first few batches are not read once again in case of
    ###
    if batch is None:
        n_iter = n_iter + 1
        continue
        
    print('\tthis batch: N_lang = %d, N_bbox = %d' %
          (batch['expr_obj1_batch'].shape[1], batch['bbox_batch'].shape[0]))

    # Forward pass
    scores_val = sess.run(scores,
        feed_dict={
            im_batch            : batch['im_batch'],
            bbox_batch          : batch['bbox_batch'],
            spatial_batch       : batch['spatial_batch'],
            text_seq_batch      : batch['text_seq_batch']
        })

    N_batch, N_box, _, _ = scores_val.shape

    # scores_val has shape [N_batch, N_box, N_box, 1]
    scores_flat = scores_val.reshape((N_batch, N_box*N_box))
    # prediction_box_ids has shape [N_batch, K] containing indices
    prediction_box_ids = np.argsort(-scores_flat, axis=1)[:, :K]  # minus to sort in descending order
    # labels has shape [N_batch, 1] containing indices
    labels = batch['label_batch'].reshape((N_batch, 1))

    is_matched = (prediction_box_ids == labels).astype(np.float32)
    is_matched_cumsum = np.cumsum(is_matched, axis=1)
    matched_ids_count = np.sum(is_matched_cumsum, axis=0)
    top_x_correct_count[:N_box*N_box] += matched_ids_count
    top_x_correct_count[N_box*N_box:] += N_batch
    total += N_batch


with open(result_file, 'w') as f:
    for k in range(K):
        f.write('recall at %d: %f (= %d / %d)\n' %
                (k+1, top_x_correct_count[k]/total, top_x_correct_count[k], total))
print('Testing results saved to %s' % result_file)
