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

# Trained model
tfmodel_file = './downloaded_models/shape_attention_weak_iter_25000.tfmodel'
vocab_file = './word_embedding/vocabulary_72700.txt'

# Data Params
eval_set = 'val'
#eval_set = 'tst'
data_filepath = './exp-shape/data/01_gen_relative_pos_%s.npz' % eval_set

# Results
result_file = './exp-shape/results/shape_attention_weak_iter_25000.%s.txt' % eval_set
visualize_dir = './exp-shape/results/shape_attention_weak_iter_25000.%s/' % eval_set
num_vis = 500

################################################################################
# Network
################################################################################

imcrop_batch = tf.placeholder(tf.float32, [N_bbox, IM_H, IM_W, 3])
spatial_batch = tf.placeholder(tf.float32, [N_bbox, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, 1])

scores = shape_model.shape_attention_net(imcrop_batch, spatial_batch, text_seq_batch,
                               num_vocab, embed_dim, lstm_dim,
                               False, False)

################################################################################
# Load data
################################################################################

# A simple data reader for shape experiment
data_file = np.load(data_filepath, encoding='bytes')
matched_pairs_list = data_file['matched_pairs_list']
query_list = data_file['query_list']
image_list = data_file['image_list']
num_images = len(image_list)

vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

imcrop_val = np.zeros((N_bbox, IM_H, IM_W, 3), np.float32)
spatial_val = np.zeros((N_bbox, 5), np.float32)
text_seq_val = np.zeros((T, 1), np.int32)
label_val = np.zeros((N_bbox, N_bbox, 1), np.float32)

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
    label_val[...] = 0
    matched_pairs = matched_pairs_list[n_iter % num_images]
    for (h1, w1), (h2, w2) in matched_pairs:
        label_val[h1*width+w1, h2*width+w2] = 1

    batch = dict(imcrop_batch=imcrop_val,
                 spatial_batch=spatial_val,
                 text_seq_batch=text_seq_val,
                 label_batch=label_val)
    return batch

################################################################################
# Load model
################################################################################

# Snapshot saver
snapshot_saver = tf.train.Saver()

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Run Initialization operations
snapshot_saver.restore(sess, tfmodel_file)

################################################################################
# Test loop
################################################################################

if not os.path.isdir(visualize_dir):
    os.mkdir(visualize_dir)
f = open(result_file, 'w')
f.write('tfmodel_file: %s\ndata_filepath: %s\nvisualize_dir: %s\n' %
        (tfmodel_file, data_filepath, visualize_dir))

num_correct, num_total = 0, 0
for n_iter in range(num_images):
    batch = load_one_batch(n_iter)

    # Forward pass
    scores_val, ((probs_obj1, probs_obj2, probs_rel),) = sess.run((scores, tf.get_collection("attention_probs")),
        feed_dict={
            imcrop_batch        : batch['imcrop_batch'],
            spatial_batch       : batch['spatial_batch'],
            text_seq_batch      : batch['text_seq_batch'],
        })

    scores_val = np.squeeze(scores_val)  # [25, 25]
    predict_idx = np.argmax(np.max(scores_val, axis=1))
    gt_idx_list = [h*width+w for ((h, w), (_, _)) in matched_pairs_list[n_iter]]
    is_correct = (predict_idx in gt_idx_list)
    num_correct += is_correct
    num_total += 1


result_str = 'test %d - correct: %d, total: %d, accuracy: %f\n' % (n_iter, num_correct, num_total, num_correct/num_total)
print(result_str, end='')
f.write(result_str)
f.close()
print('Testing results saved to %s' % result_file)
