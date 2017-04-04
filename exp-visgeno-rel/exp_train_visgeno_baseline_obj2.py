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

# Initialization Params
convnet_params = './models/convert_caffemodel/params/fasterrcnn_vgg_coco_params.npz'
wordembed_params = './word_embedding/embed_matrix.npy'

start_lr = 0.005
lr_decay_step = 160000
lr_decay_rate = 0.1
weight_decay = 0.0005
momentum = 0.95
max_iter = 360000

fix_convnet = True
vgg_dropout = False
lstm_dropout = False
vgg_lr_mult = .1

# Data Params
imdb_file = './exp-visgeno-rel/data/imdb/imdb_trn.npy'
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = visgeno_baseline_model.fastrcnn_vgg_net.channel_mean

# Snapshot Params
snapshot_interval = 10000
snapshot_file = './exp-visgeno-rel/tfmodel/visgeno_baseline_obj2_iter_%d.tfmodel'

# Log params
log_interval = 20
log_dir = './exp-visgeno-rel/tb/visgeno_baseline_obj2/'

################################################################################
# Network
################################################################################

im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, None])
label_batch = tf.placeholder(tf.int32, [None])

scores = visgeno_baseline_model.visgeno_baseline_net(im_batch, bbox_batch, spatial_batch,
    text_seq_batch, num_vocab, embed_dim, lstm_dim, vgg_dropout, lstm_dropout)

################################################################################
# Collect trainable variables, regularized variables and learning rates
################################################################################

# Fix the word embedding matrix during training
no_word_embedding_vars = [var for var in tf.trainable_variables()
                          if not var.op.name.endswith('embedding_mat')]

if fix_convnet:
    train_var_list = [var for var in no_word_embedding_vars
                      if not var.op.name.startswith('vgg_local/')]
else:
    train_var_list = no_word_embedding_vars

print('Collecting variables to train:')
for var in train_var_list: print('\t%s' % var.op.name)
print('Done.')

# Add regularization to weight matrices (excluding bias)
reg_var_list = [var for var in train_var_list
                if (var.op.name.endswith('weights') or var.op.name.endswith('Matrix'))]
print('Collecting variables for regularization:')
for var in reg_var_list: print('\t%s' % var.op.name)
print('Done.')

# Collect learning rate for trainable variables
# Use smaller learning rate for VGG_net weights
var_lr_mult = {var: (vgg_lr_mult if var.name.startswith('vgg_local') else 1.0)
               for var in train_var_list}
print('Variable learning rate multiplication:')
for var in train_var_list:
    print('\t%s: %f' % (var.op.name, var_lr_mult[var]))
print('Done.')

################################################################################
# Loss function and accuracy
################################################################################

N_lang = tf.shape(label_batch)[0]
N_bbox = tf.shape(bbox_batch)[0]
scores_flat = tf.reshape(scores, tf.convert_to_tensor([N_lang, N_bbox]))
loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=scores_flat, labels=label_batch))

loss_reg = loss.l2_regularization_loss(reg_var_list, weight_decay)
loss_total = loss_cls + loss_reg

################################################################################
# Solver
################################################################################

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_lr, global_step, lr_decay_step,
    lr_decay_rate, staircase=True)
solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
# Compute gradients
grads_and_vars = solver.compute_gradients(loss_total, var_list=train_var_list)
# Apply learning rate multiplication to gradients
grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v)
                  for g, v in grads_and_vars]
# Apply gradients
train_step = solver.apply_gradients(grads_and_vars, global_step=global_step)

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
reader = DataReader(imdb_file, vocab_file, im_mean)

################################################################################
# Snapshot and log
################################################################################

# Snapshot saver
snapshot_saver = tf.train.Saver()

# Log writer
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
loss_cls_ph = tf.placeholder(tf.float32, [])
lr_ph = tf.placeholder(tf.float32, [])
scores_ph = tf.placeholder(tf.float32, None)
tf.summary.scalar("loss_cls", loss_cls_ph)
tf.summary.scalar("lr", lr_ph)
tf.summary.histogram("scores", scores_ph)
log_step = tf.summary.merge_all()

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Run Initialization operations
sess.run(tf.global_variables_initializer())
sess.run(tf.group(*init_ops))

################################################################################
# Optimization loop
################################################################################

loss_cls_avg = 0
decay = 0.99

# Run optimization
initial_iter = sess.run(global_step)
for n_iter in range(initial_iter, max_iter):
    batch = reader.read_batch()
    print('\tthis batch: N_lang = %d, N_bbox = %d' %
          (batch['text_seq_batch'].shape[1], batch['bbox_batch'].shape[0]))

    # Forward and Backward pass
    scores_val, loss_cls_val, _, lr_val = sess.run([scores, loss_cls, train_step, learning_rate],
        feed_dict={
            im_batch            : batch['im_batch'],
            bbox_batch          : batch['bbox_batch'],
            spatial_batch       : batch['spatial_batch'],
            text_seq_batch      : batch['text_seq_batch'],
            label_batch         : batch['label_weak_obj2_batch']
        })

    # Smooth the loss and accuracy with exponential moving average
    loss_cls_avg = decay*loss_cls_avg + (1-decay)*loss_cls_val
    print('\titer = %d, loss_cls (cur) = %f, loss_cls (avg) = %f, lr = %f'
        % (n_iter, loss_cls_val, loss_cls_avg, lr_val))

    # Add to TensorBoard summary
    if (n_iter+1) % log_interval == 0 or (n_iter+1) == max_iter:
        summary = sess.run(log_step, {loss_cls_ph: loss_cls_val, lr_ph: lr_val,
                                      scores_ph: scores_val})
        log_writer.add_summary(summary, n_iter)

    # Save snapshot
    if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1), write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
