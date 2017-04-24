import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--restore_from', default='')
args = parser.parse_args()

import os; os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import time
import tensorflow as tf
import numpy as np

from models import visgeno_baseline_model
from util.cub200_train.roi_data_reader import DataReader
from util import loss
from util.eval_tools import compute_accuracy

################################################################################
# Parameters
################################################################################

# Model Params
T = 10
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000

# Initialization Params
convnet_params = './models/convert_caffemodel/params/fasterrcnn_vgg_coco_params.npz'
wordembed_params = './word_embedding/embed_matrix.npy'

# Training Params
pos_loss_mult = 15
neg_loss_mult = 0.5
softmax_label = False

weight_decay = 0.0005
max_iter = 300000

fix_convnet = False
vgg_dropout = True
lstm_dropout = False
vgg_lr_mult = 0.1

# Data Params
roidb_file = './exp-cub200/data/imdb/imdb_trainval.npy'
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = visgeno_baseline_model.fastrcnn_vgg_net.channel_mean
proposal_name = "proposals"
include_gt_bbox = True

# Snapshot Params
snapshot_interval = 10000
snapshot_file = './exp-cub200/tfmodel/cub200_det/iter_%d.tfmodel'

################################################################################
# The model
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, None])
im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 5])
shape_batch = tf.placeholder(tf.int32, [2])
if softmax_label:
    label_batch = tf.placeholder(tf.int32, [None])
else:
    label_batch = tf.placeholder(tf.float32, [None, None])

# Outputs
scores = visgeno_baseline_model.visgeno_baseline_net(im_batch, bbox_batch,
    spatial_batch, text_seq_batch, num_vocab, embed_dim, lstm_dim,
    vgg_dropout=vgg_dropout, lstm_dropout=lstm_dropout)

scores = tf.reshape(scores, shape_batch)

################################################################################
# Collect trainable variables, regularized variables and learning rates
################################################################################

if fix_convnet:
    train_var_list = [var for var in tf.trainable_variables()
                      if not var.name.startswith('vgg_local/')]
else:
    # Only train the fc layers of convnet and keep conv layers fixed
    train_var_list = [var for var in tf.trainable_variables()
                      if not var.name.startswith('vgg_local/conv')]
print('Collecting variables to train:')
for var in train_var_list: print('\t%s' % var.name)
print('Done.')

# Add regularization to weight matrices (excluding bias)
reg_var_list = [v for v in train_var_list
                if v.op.name.endswith('weights')]
print('Collecting variables for regularization:')
for var in reg_var_list: print('\t%s' % var.name)
print('Done.')

# Collect learning rate for trainable variables
var_lr_mult = {var: (vgg_lr_mult if var.name.startswith('vgg_local') else 1.0)
               for var in train_var_list}
print('Variable learning rate multiplication:')
for var in train_var_list:
    print('\t%s: %f' % (var.name, var_lr_mult[var]))
print('Done.')

################################################################################
# Loss function and accuracy
################################################################################

if softmax_label:
    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_batch))
else:
    cls_loss = loss.weighed_logistic_loss(scores, label_batch, pos_loss_mult, neg_loss_mult)
reg_loss = loss.l2_regularization_loss(reg_var_list, weight_decay)
total_loss = cls_loss + reg_loss

################################################################################
# Solver
################################################################################

solver = tf.train.AdamOptimizer()
# Compute gradients
grads_and_vars = solver.compute_gradients(total_loss, var_list=train_var_list)
# Apply learning rate multiplication to gradients
grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v)
                  for g, v in grads_and_vars]
# Apply gradients
train_step = solver.apply_gradients(grads_and_vars)

################################################################################
# Initialize parameters and load data
################################################################################

init_ops = []
# Initialize CNN Parameters
convnet_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                  'conv3_1', 'conv3_2', 'conv3_3',
                  'conv4_1', 'conv4_2', 'conv4_3',
                  'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7']
processed_params = np.load(convnet_params)
processed_W = processed_params['processed_W'][()]
processed_B = processed_params['processed_B'][()]
with tf.variable_scope('vgg_local', reuse=True):
    for l_name in convnet_layers:
        assign_W = tf.assign(tf.get_variable(l_name + '/weights'), processed_W[l_name])
        assign_B = tf.assign(tf.get_variable(l_name + '/biases'), processed_B[l_name])
        init_ops += [assign_W, assign_B]
processed_params.close()

# Initialize word embedding matrix
embedding_mat_val = np.load(wordembed_params)
with tf.variable_scope('lstm', reuse=True):
    embedding_mat = tf.get_variable("embedding_mat", [num_vocab, embed_dim])
    init_we = tf.assign(embedding_mat, embedding_mat_val)

init_ops += [init_we]

# Load data
reader = DataReader(roidb_file, vocab_file, im_mean, proposal_name, T=T,
    include_gt_bbox=include_gt_bbox, softmax_label=softmax_label)

snapshot_saver = tf.train.Saver(max_to_keep=None)
sess = tf.Session()

# Run Initialization operations
if args.restore_from:
    snapshot_saver.restore(sess, args.restore_from)
else:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.group(*init_ops))

################################################################################
# Optimization loop
################################################################################

cls_loss_avg = 0
avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
decay = 0.99

# Run optimization
time0 = time.time()
for n_iter in range(max_iter):
    # Read one batch
    batch = reader.read_batch()
    text_seq_val  = batch['text_seq_batch']
    im_val        = batch['im_batch']
    bbox_val      = batch['bbox_batch']
    spatial_val   = batch['spatial_batch']
    label_val     = batch['label_batch']

    print('\tthis batch: image height %d, width %d, with %d sentences x %d proposal boxes = %d scores' %
          (im_val.shape[1], im_val.shape[2], text_seq_val.shape[1], bbox_val.shape[0],
           text_seq_val.shape[1]*bbox_val.shape[0]))

    # Forward and Backward pass
    scores_val, cls_loss_val, _ = sess.run([scores, cls_loss, train_step],
        feed_dict={
            text_seq_batch  : text_seq_val,
            im_batch        : im_val,
            bbox_batch      : bbox_val,
            spatial_batch   : spatial_val,
            label_batch     : label_val,
            shape_batch     : [text_seq_val.shape[1], bbox_val.shape[0]]
        })
    cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
    print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f'
        % (n_iter, cls_loss_val, cls_loss_avg))

    # Accuracy
    if not softmax_label:
        accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
        avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
        avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
        print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
              % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
        print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
              % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

    if (n_iter % 100) == 0:
        time_spent = time.time() - time0
        print('\n\tTotal time elapsed: %f sec. Average time per batch: %f sec\n' %
              (time_spent, time_spent / (n_iter+1)))

    # Save snapshot
    if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1), write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
sess.close()
