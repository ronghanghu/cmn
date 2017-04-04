from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

def weighed_logistic_loss(scores, labels, pos_loss_mult=1.0, neg_loss_mult=1.0):
    # Apply different weights to loss of positive samples and negative samples
    # positive samples have label 1 while negative samples have label 0
    # Classification loss as the average of weighed per-score loss
    cls_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        scores, labels, pos_loss_mult/neg_loss_mult)) * neg_loss_mult

    return cls_loss

def top1_ranking_loss(scores, labels, margin=0):
    # input scores and labels has same shape as in
    # tf.nn.sparse_softmax_cross_entropy_with_logits
    # loss is \sum_i max(0, margin + score[i] - score[label])

    # we want to do scores[:, labels], but that's not available in tensorflow
    # so we transpose and get scores_t[labels]
    scores_t = tf.transpose(scores)
    gt_scores = tf.reshape(tf.gather(scores_t, labels), [-1, 1])
    loss_sum = tf.reduce_sum(tf.nn.relu(margin + scores - gt_scores))
    # average within a batch
    loss_avg = tf.div(loss_sum, tf.cast(tf.shape(scores)[0], tf.float32))
    return loss_avg

def l2_regularization_loss(variables, weight_decay):
    l2_losses = [tf.nn.l2_loss(var) for var in variables]
    total_l2_loss = weight_decay * tf.add_n(l2_losses)
    return total_l2_loss
