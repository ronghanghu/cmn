from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # using GPU 0

import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform

from models import shape_model, spatial_feat, vgg_net
from util import loss, eval_tools, text_processing

################################################################################
# Parameters
################################################################################

# Model Params
N_bbox = 25  # total number of bounding box per image
T = 20
IM_H = 224
IM_W = 224
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000

# Initialization Params
convnet_params = './models/convert_caffemodel/params/vgg_params.npz'

# Training Params
# If set to True, use strong supervision with paired labels (obj1, obj2)
# Otherwise, only use weak supervision with only labels (obj1)
strong_supervision = False

start_lr = 0.005
lr_decay_step = 10000
lr_decay_rate = 0.1
weight_decay = 0.0005
momentum = 0.95
max_iter = 25000

fix_convnet = True
vgg_dropout = False
lstm_dropout = False
vgg_lr_mult = .1

# Data Params
data_filepath = './exp-shape/data/01_gen_relative_pos_trn.npz'
vocab_file = './word_embedding/vocabulary_72700.txt'

# Snapshot Params
snapshot_interval = 5000
if strong_supervision:
    snapshot_file = './exp-shape/tfmodel/shape_attention_strong_iter_%d.tfmodel'
else:
    snapshot_file = './exp-shape/tfmodel/shape_attention_weak_iter_%d.tfmodel'

################################################################################
# Network
################################################################################

imcrop_batch = tf.placeholder(tf.float32, [N_bbox, IM_H, IM_W, 3])
spatial_batch = tf.placeholder(tf.float32, [N_bbox, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, 1])
label_batch = tf.placeholder(tf.int32, [1])

scores = shape_model.shape_attention_net(imcrop_batch, spatial_batch,
                               text_seq_batch, num_vocab, embed_dim, lstm_dim,
                               vgg_dropout, lstm_dropout)
scores = tf.squeeze(scores, squeeze_dims=[0, 3])
# max-pool the score along the second object in weak supervision scenario
if strong_supervision:
    scores = tf.reshape(scores, [1, N_bbox*N_bbox])
else:
    scores = tf.reduce_max(scores, [1])
    scores = tf.reshape(scores, [1, N_bbox])

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

loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=scores, labels=label_batch))
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

################################################################################
# Load data
################################################################################

# A simple data reader for shape experiment
data_file = np.load(data_filepath, encoding='bytes')
query_list = data_file['query_list']
matched_pairs_list = data_file['matched_pairs_list']
image_list = data_file['image_list']
num_images = len(image_list)

vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

imcrop_val = np.zeros((N_bbox, IM_H, IM_W, 3), np.float32)
spatial_val = np.zeros((N_bbox, 5), np.float32)
text_seq_val = np.zeros((T, 1), np.int32)
label_val = np.zeros(1, np.float32)

height, width = 5, 5
bboxes = np.zeros((N_bbox, 4))

def load_one_batch(n_iter):
    global imcrop_val
    global spatial_val
    global text_seq_val
    global label_val

    print('data reader: epoch = %d, batch = %d / %d' %
          (n_iter // num_images, n_iter % num_images, num_images))

    # Read one batch
    # Get images
    image = image_list[n_iter % num_images]
    for h in range(height):
        for w in range(width):
            crop = image[h*10:(h+1)*10, w*10:(w+1)*10, :]
            imcrop_val[h*width+w] = skimage.transform.resize(crop, [IM_H, IM_W])
            bboxes[h*width+w] = [w, h, w+1, h+1] # [x1, y1, x2, y2]
    imcrop_val *= 255
    imcrop_val -= vgg_net.channel_mean

    # Get spatial batch
    spatial_val = spatial_feat.spatial_feature_from_bbox(bboxes, im_h=height, im_w=width)

    # Get text sequence
    expr_obj = query_list[n_iter % num_images]
    text_seq_val[:, 0] = text_processing.preprocess_sentence(expr_obj, vocab_dict, T)

    # Get labels
    matched_pairs = matched_pairs_list[n_iter % num_images]
    (h1, w1), (h2, w2) = matched_pairs[0]  # just take the first matched_pair
    if strong_supervision:
        label_val[...] = (h1*width+w1)*N_bbox + (h2*width+w2)
    else:
        label_val[...] = (h1*width+w1)

    batch = dict(imcrop_batch=imcrop_val,
                 spatial_batch=spatial_val,
                 text_seq_batch=text_seq_val,
                 label_batch=label_val)
    return batch

################################################################################
# Snapshot
################################################################################

# Snapshot saver
snapshot_saver = tf.train.Saver()

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Run Initialization operations
sess.run(tf.global_variables_initializer())
sess.run(tf.group(*init_ops))

################################################################################
# Optimization loop
################################################################################

loss_cls_avg = 0
accuracy_avg = 0
decay = 0.99

# Run optimization
for n_iter in range(max_iter):
    batch = load_one_batch(n_iter)

    # Forward and Backward pass
    scores_val, loss_cls_val, _, lr_val = sess.run([scores, loss_cls, train_step, learning_rate],
        feed_dict={
            imcrop_batch        : batch['imcrop_batch'],
            spatial_batch       : batch['spatial_batch'],
            text_seq_batch      : batch['text_seq_batch'],
            label_batch         : batch['label_batch']
        })

    predicts = np.argmax(scores_val)
    labels = batch['label_batch']
    print('predict:', predicts, 'ground-truth:', labels)
    accuracy = np.mean(predicts == labels)
    accuracy_avg = decay*accuracy_avg + (1-decay)*accuracy
    print('\titer = %d, accuracy (cur) = %f, accuracy (avg) = %f'
          % (n_iter, accuracy, accuracy_avg))

    # Smooth the loss and accuracy with exponential moving average
    loss_cls_avg = decay*loss_cls_avg + (1-decay)*loss_cls_val
    print('\titer = %d, loss_cls (cur) = %f, loss_cls (avg) = %f, lr = %f'
        % (n_iter, loss_cls_val, loss_cls_avg, lr_val))

    # Save snapshot
    if (n_iter+1) % snapshot_interval == 0 or (n_iter+1) == max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1), write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
