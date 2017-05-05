from __future__ import absolute_import, division, print_function

import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
import skimage.io
import colorsys

import detection_demo

# Model path
pretrained_model = './explanation_classifier_model/00000016'
vocab_file = './data/vocab.txt'

with open(vocab_file) as f:
    vocab_list = [l.lower().strip() for l in f.readlines()]
num_vocab = len(vocab_list)
word2vocab_idx = {v: n_v for n_v, v in enumerate(vocab_list)}
print('number of words in vocab_list:', num_vocab)

import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def tokenize(sentence):
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def noun_phrases_to_bow(phrase, vocab_list):
    '''
    Construct bag-of-words features from a phrase
    '''
    bow = np.zeros(len(vocab_list), np.float32)
    words = tokenize(phrase)
    for w in words:
        if w in word2vocab_idx:
            bow[word2vocab_idx[w]] += 1
    return bow

# old set of variables, before this module being imported
_existing_vars = set(tf.global_variables())

feature_hidden_dim = 512
lstm_units = 512
classifier_hidden_dim = 256
bow_dim = 2632
visfeat_dim = 4096+8

T = 15

# the model
seq_length_batch = tf.placeholder(tf.int32, [None])
bow_batch = tf.placeholder(tf.float32, [T, None, bow_dim])
visfeat_batch = tf.placeholder(tf.float32, [T, None, visfeat_dim])
bbox_score_batch = tf.placeholder(tf.float32, [T, None, 1])

def explanation_classification_model(bow_batch, visfeat_batch, bbox_score_batch, seq_length_batch,
    feature_hidden_dim, lstm_units, classifier_hidden_dim,
    scope='explanation_classifier', reuse=None):
    
    # concatenate all the features, and
    # use a fully-connected layer to map the features to a new dimension
    all_features = tf.concat([bow_batch, visfeat_batch, bbox_score_batch], axis=-1)
    all_features = tf.reshape(all_features, [-1, all_features.get_shape().as_list()[-1]])
    all_features_mapped = tf.layers.dense(all_features, feature_hidden_dim, activation=tf.nn.relu)
    all_features_mapped = tf.reshape(all_features_mapped, [T, -1, feature_hidden_dim])
    
    # feed the features into a LSTM
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    _, state = tf.nn.dynamic_rnn(cell, all_features_mapped, sequence_length=seq_length_batch,
                             dtype=tf.float32, time_major=True)
    
    # the final classifier: a two-layer network
    embeddings = state.h
    embeddings_mapped = tf.layers.dense(embeddings, classifier_hidden_dim, activation=tf.nn.relu)
    scores = tf.layers.dense(embeddings_mapped, 1)
    scores = tf.reshape(scores, [-1])
    return scores

scores = explanation_classification_model(bow_batch, visfeat_batch,
    bbox_score_batch, seq_length_batch, feature_hidden_dim, lstm_units, classifier_hidden_dim)

# variables in this module
module_vars = [v for v in tf.global_variables() if v not in _existing_vars]

# Load model
snapshot_saver = tf.train.Saver(module_vars)
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
snapshot_saver.restore(sess, pretrained_model)

def run_demo(im_path, raw_query, save_path):
    # Step 0: split the raw query string into the explanation and the noun phrase
    raw_query_split = raw_query.split(';')
    explanation = raw_query_split[0]
    query_list = [q.strip() for q in raw_query_split[1:] if len(q.strip()) > 0]
    print('explanation sentence:', explanation)
    print('noun phrases (got %d):' % len(query_list))
    for n_q, q in enumerate(query_list):
        print('%d:'%(n_q+1), q)
    
    # Step 1: extract the bounding box prediction and image feature for each example
    proposal_bboxes, scores_val, im_visfeat, bbox_visfeat = \
        detection_demo.run_on_image(im_path, query_list)
    scores_val = scores_val.reshape((len(query_list), len(proposal_bboxes)))
    
    pred_bbox_inds = np.argmax(scores_val, axis=1)
    query_scores = [scores_val[n_q, pred_bbox_inds[n_q]] for n_q in range(len(query_list))]
    query_visfeat = list(bbox_visfeat[pred_bbox_inds])
    pred_bboxes = proposal_bboxes[pred_bbox_inds]
    
    query_dict = {query_list[n_q]: {'score': query_scores[n_q],
                                    'visfeat': query_visfeat[n_q]}
                  for n_q in range(len(query_list))}
    
    # Step 1.1: plot results on image
    im = skimage.io.imread(im_path)
    if im.ndim == 2:
        im = np.tile(im[..., np.newaxis], (1, 1, 3))
    im_vis = detection_demo.plot_bboxes_on_im(im, pred_bboxes)
    skimage.io.imsave(save_path, im_vis)
    
    # Step 2: Build the input batch for the classifier
    N = 1
    seq_length_array = np.zeros(N, np.int32)    
    bow_array = np.zeros((T, N, num_vocab), np.float32)
    visfeat_array = np.zeros((T, N, 4096 + 8), np.float32)
    bbox_score_array = np.zeros((T, N, 1), np.float32)
    
    n = 0
    seq_length_array[n] = 1 + len(query_list)
        
    # Put the whole sentence feature at the beginning
    bow_array[0, n] = noun_phrases_to_bow(explanation, vocab_list)
    visfeat_array[0, n] = im_visfeat
    bbox_score_array[0, n] = 0  # assign 0 as scores of every sentence

    for t, phrase in enumerate(query_list):
        bow_array[t+1, n] = noun_phrases_to_bow(phrase, vocab_list)
        visfeat_array[t+1, n] = query_dict[phrase]['visfeat']
        bbox_score_array[t+1, n] = query_dict[phrase]['score']
        
    # Step 3: run forward pass in the classifier to get the score
    scores_val = sess.run(scores,
                          {seq_length_batch: seq_length_array,
                           bow_batch: bow_array,
                           visfeat_batch: visfeat_array,
                           bbox_score_batch: bbox_score_array})
    scores_val = scores_val.reshape(())  # reshape to scalar
    
    return scores_val

