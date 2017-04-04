from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # using GPU 0

import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform

from models import visual7w_baseline_model, spatial_feat, fastrcnn_vgg_net
from util.visual7w_baseline_train.visual7w_baseline_data_reader import DataReader
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
imdb_file = './exp-visual7w/data/imdb/imdb_tst.npy'
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = visual7w_baseline_model.fastrcnn_vgg_net.channel_mean

# Snapshot Params
snapshot_file = './downloaded_models/visual7w_baseline_iter_200000.tfmodel'

################################################################################
# Network
################################################################################

im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, None])

scores = visual7w_baseline_model.visual7w_baseline_net(im_batch, bbox_batch, spatial_batch,
    text_seq_batch, num_vocab, embed_dim, lstm_dim, vgg_dropout=False, lstm_dropout=False)
num_choices = 4
scores = tf.reshape(scores, [-1, num_choices])

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
    batch = reader.read_batch()

    # Forward and Backward pass
    scores_val = sess.run(scores,
        feed_dict={
            im_batch            : batch['im_batch'],
            bbox_batch          : batch['bbox_batch'],
            spatial_batch       : batch['spatial_batch'],
            text_seq_batch      : batch['text_seq_batch']
        })

    predicts = np.argmax(scores_val, axis=1)
    labels = batch['label_batch']
    num_correct += np.sum(predicts == labels)
    num_total += len(labels)

    print('\titer = %d, accuracy (avg) = %f' % (n_iter, num_correct / num_total))

print('On the following imdb:', imdb_file)
print('Using the following snapshot:', snapshot_file)
print('final accuracy: %f (= %d / %d)' % (num_correct / num_total, num_correct, num_total))
