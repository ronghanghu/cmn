from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from util.rnn import lstm_layer as lstm
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from tensorflow.python.ops.nn import dropout as drop

def lstm_encoder(text_seq_batch, name, num_vocab, embed_dim, lstm_dim,
             apply_dropout, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        embedding_mat = tf.get_variable("embedding_mat", [num_vocab, embed_dim])
        # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)

        # Take the output at the final timestep of LSTM.
        lstm_top = lstm("lstm_lang", embedded_seq, None, output_dim=lstm_dim,
                        num_layers=1, forget_bias=1.0,
                        apply_dropout=apply_dropout, concat_output=False)[-1]

    return lstm_top

def attbilstm(text_seq_batch, name, num_vocab, embed_dim, lstm_dim,
    apply_dropout, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        T = tf.shape(text_seq_batch)[0]
        N = tf.shape(text_seq_batch)[1]

        # 0. Word embedding
        embedding_mat = tf.get_variable("embedding_mat", [num_vocab, embed_dim])
        # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)

        # 1. Encode the sentence into a vector representation, using the final
        # hidden states in a two-layer bidirectional LSTM network
        seq_length = tf.ones(to_T([N]), dtype=tf.int32)*T
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=True)
        outputs1_raw, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell,
            embedded_seq, seq_length, dtype=tf.float32, time_major=True,
            scope="bidirectional_lstm1")
        outputs1 = tf.concat(outputs1_raw, axis=2)
        outputs2_raw, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell,
            outputs1, seq_length, dtype=tf.float32, time_major=True,
            scope="bidirectional_lstm2")
        outputs2 = tf.concat(outputs2_raw, axis=2)
        # q_reshape has shape [T, N, lstm_dim*4]
        q_reshape = tf.concat([outputs1, outputs2], axis=2)
        if apply_dropout:
            q_reshape = drop(q_reshape, 0.5)

        # 2. three attention units over the words in each sentence
        with tf.variable_scope("attention"):
            q_reshape_flat = tf.reshape(q_reshape, to_T([T*N, lstm_dim*4]))

            score_shape = to_T([T, N, 1])
            scores_obj1 = tf.reshape(fc('fc_scores_obj1', q_reshape_flat, output_dim=1), score_shape)
            scores_obj2 = tf.reshape(fc('fc_scores_obj2', q_reshape_flat, output_dim=1), score_shape)
            scores_rel = tf.reshape(fc('fc_scores_rel', q_reshape_flat, output_dim=1), score_shape)

            # 2.4 Compute probability and average BoW representation
            # probs_obj1, probs_obj2 and probs_rel has shape [T, N, 1]
            # Remove the probability over <pad> (<pad> is 0)
            is_not_pad = tf.cast(tf.not_equal(text_seq_batch, 0)[..., tf.newaxis], tf.float32)
            probs_obj1 = tf.nn.softmax(scores_obj1, dim=0)*is_not_pad
            probs_obj2 = tf.nn.softmax(scores_obj2, dim=0)*is_not_pad
            probs_rel = tf.nn.softmax(scores_rel, dim=0)*is_not_pad
            probs_obj1 = probs_obj1 / tf.reduce_sum(probs_obj1, 0, keep_dims=True)
            probs_obj2 = probs_obj2 / tf.reduce_sum(probs_obj2, 0, keep_dims=True)
            probs_rel = probs_rel / tf.reduce_sum(probs_rel, 0, keep_dims=True)

            tf.add_to_collection("attention_probs", (probs_obj1, probs_obj2, probs_rel))

            # BoW_obj1, BoW_obj2 and BoW_rel has shape [N, embed_dim]
            BoW_obj1 = tf.reduce_sum(probs_obj1*embedded_seq, reduction_indices=0)
            BoW_obj2 = tf.reduce_sum(probs_obj2*embedded_seq, reduction_indices=0)
            BoW_rel = tf.reduce_sum(probs_rel*embedded_seq, reduction_indices=0)
            BoW_obj1.set_shape([None, embed_dim])
            BoW_obj2.set_shape([None, embed_dim])
            BoW_rel.set_shape([None, embed_dim])

    return (BoW_obj1, BoW_obj2, BoW_rel)

def attbilstm_simple(text_seq_batch, name, num_vocab, embed_dim,
                     lstm_dim, apply_dropout, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        T = tf.shape(text_seq_batch)[0]
        N = tf.shape(text_seq_batch)[1]

        # 0. Word embedding
        embedding_mat = tf.get_variable("embedding_mat", [num_vocab, embed_dim])
        # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)

        # 1. Encode the sentence into a vector representation, using the final
        # hidden states in a bidirectional LSTM network
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=True)
        seq_length = tf.ones(to_T([N]), dtype=tf.int32)*T
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell,
            embedded_seq, seq_length, dtype=tf.float32, time_major=True,
            scope="bidirectional_stm")
        q_reshape = tf.concat(outputs, axis=2)
        if apply_dropout:
            q_reshape = drop(q_reshape, 0.5)

        # 2. three attention units over the words in each sentence
        with tf.variable_scope("attention"):
            # 2.1 Map the word embedding vectors to the same dimension as q
            embedded_seq_reshape = tf.reshape(embedded_seq, [-1, embed_dim])
            word_seq_embed = fc('attention_embed', embedded_seq_reshape, output_dim=lstm_dim*2)
            word_seq_embed = tf.reshape(word_seq_embed, to_T([T, N, lstm_dim*2]))

            # 2.2 Elementwise multiply word_seq_embed with q and l2-normalization
            eltwise_mult = tf.nn.l2_normalize(word_seq_embed * q_reshape, 2)

            # 2.3 Classification for attention scores
            eltwise_mult = tf.reshape(eltwise_mult, [-1, lstm_dim*2])
            # scores_obj1, scores_obj2 and scores_rel has shape [T, N, 1]
            score_shape = to_T([T, N, 1])
            scores_obj1 = tf.reshape(fc('fc_scores_obj1', eltwise_mult, output_dim=1), score_shape)
            scores_obj2 = tf.reshape(fc('fc_scores_obj2', eltwise_mult, output_dim=1), score_shape)
            scores_rel = tf.reshape(fc('fc_scores_rel', eltwise_mult, output_dim=1), score_shape)

            # 2.4 Compute probability and average BoW representation
            # probs_obj1, probs_obj2 and probs_rel has shape [T, N, 1]
            # Remove the probability over <pad> (<pad> is 0)
            is_not_pad = tf.cast(tf.not_equal(text_seq_batch, 0)[..., tf.newaxis], tf.float32)
            probs_obj1 = tf.nn.softmax(scores_obj1, dim=0)*is_not_pad
            probs_obj2 = tf.nn.softmax(scores_obj2, dim=0)*is_not_pad
            probs_rel = tf.nn.softmax(scores_rel, dim=0)*is_not_pad
            probs_obj1 = probs_obj1 / tf.reduce_sum(probs_obj1, 0, keep_dims=True)
            probs_obj2 = probs_obj2 / tf.reduce_sum(probs_obj2, 0, keep_dims=True)
            probs_rel = probs_rel / tf.reduce_sum(probs_rel, 0, keep_dims=True)

            # BoW_obj1, BoW_obj2 and BoW_rel has shape [N, embed_dim]
            BoW_obj1 = tf.reduce_sum(probs_obj1*embedded_seq, reduction_indices=0)
            BoW_obj2 = tf.reduce_sum(probs_obj2*embedded_seq, reduction_indices=0)
            BoW_rel = tf.reduce_sum(probs_rel*embedded_seq, reduction_indices=0)
            BoW_obj1.set_shape([None, embed_dim])
            BoW_obj2.set_shape([None, embed_dim])
            BoW_rel.set_shape([None, embed_dim])

            tf.add_to_collection("attention_probs", (probs_obj1, probs_obj2, probs_rel))

    return (BoW_obj1, BoW_obj2, BoW_rel)
