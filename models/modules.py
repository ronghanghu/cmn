from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from util.cnn import fc_layer as fc

def localization_module(vis_feat, spatial_feat, lang_feat,
    scope="localization_module", reuse=None):
    # Input:
    #   vis_feat: [N, D_vis]
    #   spatial_feat: [N, D_spatial]
    #   lang_feat: [N, D_lang]
    # Output:
    #   localization_scores: [N, 1]
    #
    # This function is not responsible for initializing the variables. Please
    # handle variable initialization outside.

    with tf.variable_scope(scope, reuse=reuse):
        # An embedding module that maps the visual feature plus the spatial feature
        # linearly to the same dimension as the language feature
        D_lang = lang_feat.get_shape().as_list()[-1]
        vis_spatial_feat = tf.concat([vis_feat, spatial_feat], axis=1)
        vis_spatial_embed = fc('vis_spatial_embed', vis_spatial_feat, output_dim=D_lang)

        # Elementwise multiplication with language feature and l2-normalization
        eltwise_mult = tf.nn.l2_normalize(vis_spatial_embed * lang_feat, 1)

        # Localization scores as linear classification over the l2-normalized
        localization_scores = fc('localization_scores', eltwise_mult, output_dim=1)

    return localization_scores

def localization_module_grid_score(vis_feat, spatial_feat, lang_feat,
    scope="localization_module", reuse=None):
    # Input:
    #   vis_feat: [N_vis, D_vis]
    #   spatial_feat: [N_vis, D_spatial]
    #   lang_feat: [N_lang, D_lang]
    # Output:
    #   localization_scores: [N_lang, N_vis, 1]
    #
    # This function is not responsible for initializing the variables. Please
    # handle variable initialization outside.

    with tf.variable_scope(scope, reuse=reuse):
        # An embedding module that maps the visual feature plus the spatial feature
        # linearly to the same dimension as the language feature
        N_lang = tf.shape(lang_feat)[0]
        D_lang = lang_feat.get_shape().as_list()[-1]
        vis_spatial_feat = tf.concat([vis_feat, spatial_feat], axis=1)
        vis_spatial_embed = fc('vis_spatial_embed', vis_spatial_feat, output_dim=D_lang)

        # Reshape visual feature and language feature for broadcast multiplication
        lang_feat = tf.reshape(lang_feat, [-1, 1, D_lang])
        vis_spatial_embed = tf.reshape(vis_spatial_embed, [1, -1, D_lang])

        # Elementwise multiplication with language feature and l2-normalization
        eltwise_mult = tf.nn.l2_normalize(vis_spatial_embed * lang_feat, 2)
        eltwise_mult = tf.reshape(eltwise_mult, [-1, D_lang])

        # Localization scores as linear classification over the l2-normalized
        localization_scores = fc('localization_scores', eltwise_mult, output_dim=1)
        localization_scores = tf.reshape(localization_scores, to_T([N_lang, -1, 1]))

    return localization_scores

def localization_module_batch_score(vis_feat, spatial_feat, lang_feat,
    scope="localization_module", reuse=None):
    # Input:
    #   vis_feat: [N_batch, N_vis, D_vis]
    #   spatial_feat: [N_batch, N_vis, D_spatial]
    #   lang_feat: [N_batch, D_lang]
    # Output:
    #   localization_scores: [N_batch, N_vis, 1]
    #
    # This function is not responsible for initializing the variables. Please
    # handle variable initialization outside.

    with tf.variable_scope(scope, reuse=reuse):
        # An embedding module that maps the visual feature plus the spatial feature
        # linearly to the same dimension as the language feature
        N_batch = tf.shape(vis_feat)[0]
        N_vis = tf.shape(vis_feat)[1]
        D_vis = vis_feat.get_shape().as_list()[-1]
        D_spatial = spatial_feat.get_shape().as_list()[-1]
        D_lang = lang_feat.get_shape().as_list()[-1]

        # flatten the visual and spatial features and embed them to the same
        # dimension as the language feature
        vis_spatial_feat = tf.concat([vis_feat, spatial_feat], axis=2)
        vis_spatial_feat = tf.reshape(vis_spatial_feat, [-1, D_vis+D_spatial])
        vis_spatial_embed = fc('vis_spatial_embed', vis_spatial_feat, output_dim=D_lang)

        # Reshape visual feature and language feature for broadcast multiplication
        lang_feat = tf.reshape(lang_feat, [-1, 1, D_lang])
        vis_spatial_embed = tf.reshape(vis_spatial_embed, to_T([N_batch, -1, D_lang]))

        # Elementwise multiplication with language feature and l2-normalization
        eltwise_mult = tf.nn.l2_normalize(vis_spatial_embed * lang_feat, 2)
        eltwise_mult = tf.reshape(eltwise_mult, [-1, D_lang])

        # Localization scores as linear classification over the l2-normalized
        localization_scores = fc('localization_scores', eltwise_mult, output_dim=1)
        localization_scores = tf.reshape(localization_scores, to_T([N_batch, N_vis, 1]))

    return localization_scores

def relationship_module_spatial_only(spatial_feat1, scores1,
                                     spatial_feat2, scores2, lang_feat,
                                     scope="relationship_module_spatial_only",
                                     reuse=None):
    # Input shape:
    #   spatial_feat1, spatial_feat2 : [N1, D_spatial], [N2, D_spatial]
    #   scores1, scores2: [N1, 1], [N2, 1]
    #   lang_feat: [1, D_lang]
    # Output shape:
    #   relationship_scores: [N1, N2, 1]
    #
    # This function is not responsible for initializing the variables. Please
    # handle variable initialization outside.

    with tf.variable_scope(scope, reuse=reuse):
        # An embedding module that maps the visual feature plus the spatial feature
        # linearly to the same dimension as the language feature
        D_lang = lang_feat.get_shape().as_list()[-1]

        N1 = tf.shape(spatial_feat1)[0]
        N2 = tf.shape(spatial_feat2)[0]

        D_spatial = spatial_feat1.get_shape().as_list()[-1]

        # Tiled spatial features of size [N1, N2, 5*2], such that
        # spatial_feat_tiled[i, j] = [ spatial_feat1[i], spatial_feat1[j] ]
        spatial_feat_tiled = tf.reshape(tf.concat([
            tf.tile(tf.reshape(spatial_feat1, [-1, 1, D_spatial]), to_T([1, N2, 1])),
            tf.tile(tf.reshape(spatial_feat2, [1, -1, D_spatial]), to_T([N1, 1, 1]))
        ], axis=2), [-1, D_spatial*2])

        spatial_embed = fc('spatial_embed', spatial_feat_tiled, output_dim=D_lang)

        # Elementwise multiplication with language feature and l2-normalization
        eltwise_mult = tf.nn.l2_normalize(spatial_embed * lang_feat, 1)

        # Localization scores as linear classification over the l2-normalized
        relationship_scores = fc('relationship_scores', eltwise_mult, output_dim=1)
        relationship_scores = tf.reshape(relationship_scores, to_T([N1, N2, 1]))

        final_scores = tf.add(tf.add(tf.reshape(scores1, [-1, 1, 1]),
                                     tf.reshape(scores2, [1, -1, 1])),
                              relationship_scores)
        final_scores.set_shape([None, None, 1])

    return final_scores

def relationship_module_spatial_only_grid_score(spatial_feat1, scores1,
                                     spatial_feat2, scores2, lang_feat,
                                     scope="relationship_module_spatial_only",
                                     rescale_scores=False, reuse=None):
    # Input shape:
    #   spatial_feat1, spatial_feat2 : [N1, D_spatial], [N2, D_spatial]
    #   scores1, scores2: [N_lang, N1, 1], [N_lang, N2, 1]
    #   lang_feat: [N_lang, D_lang]
    # Output shape:
    #   relationship_scores: [N_lang, N1, N2, 1]
    #
    # This function is not responsible for initializing the variables. Please
    # handle variable initialization outside.

    with tf.variable_scope(scope, reuse=reuse):
        # An embedding module that maps the visual feature plus the spatial feature
        # linearly to the same dimension as the language feature
        N_lang = tf.shape(lang_feat)[0]
        D_lang = lang_feat.get_shape().as_list()[-1]

        N1 = tf.shape(spatial_feat1)[0]
        N2 = tf.shape(spatial_feat2)[0]

        D_spatial = spatial_feat1.get_shape().as_list()[-1]

        # Tiled spatial features of size [N1, N2, 5*2], such that
        # spatial_feat_tiled[i, j] = [ spatial_feat1[i], spatial_feat1[j] ]
        spatial_feat_tiled = tf.reshape(tf.concat([
            tf.tile(tf.reshape(spatial_feat1, [-1, 1, D_spatial]), to_T([1, N2, 1])),
            tf.tile(tf.reshape(spatial_feat2, [1, -1, D_spatial]), to_T([N1, 1, 1]))
        ], axis=2), [-1, D_spatial*2])

        # Embedded spatial feature of size [N1xN2, D_lang]
        spatial_embed = fc('spatial_embed', spatial_feat_tiled, output_dim=D_lang)

        # Elementwise multiplication with language feature and l2-normalization
        eltwise_mult = tf.nn.l2_normalize(tf.reshape(spatial_embed, [1, -1, D_lang]) *
                                          tf.reshape(lang_feat, [-1, 1, D_lang]), 2)
        eltwise_mult = tf.reshape(eltwise_mult, [-1, D_lang])

        # Localization scores as linear classification over the l2-normalized
        relationship_scores = fc('relationship_scores', eltwise_mult, output_dim=1)
        relationship_scores = tf.reshape(relationship_scores, to_T([N_lang, N1, N2, 1]))
        # Rescale the scores, if specified
        if rescale_scores:
            alpha_obj1 = tf.get_variable("alpha_obj1", shape=[], dtype=tf.float32,
                                         initializer=tf.constant_initializer(1))
            alpha_obj2 = tf.get_variable("alpha_obj2", shape=[], dtype=tf.float32,
                                         initializer=tf.constant_initializer(1))
            alpha_rel = tf.get_variable("alpha_rel", shape=[], dtype=tf.float32,
                                         initializer=tf.constant_initializer(1))
            scores1 = tf.multiply(scores1, alpha_obj1)
            scores2 = tf.multiply(scores2, alpha_obj2)
            relationship_scores = tf.multiply(relationship_scores, alpha_rel)

        final_scores = tf.add(tf.add(tf.reshape(scores1, to_T([N_lang, N1, 1, 1])),
                                     tf.reshape(scores2, to_T([N_lang, 1, N2, 1]))),
                              relationship_scores)
        final_scores.set_shape([None, None, None, 1])

    return final_scores

def relationship_module_spatial_only_batch_score(spatial_feat1, scores1,
                                     spatial_feat2, scores2, lang_feat,
                                     scope="relationship_module_spatial_only",
                                     rescale_scores=False, reuse=None):
    # Input shape:
    #   spatial_feat1, spatial_feat2 : [N_batch, N1, D_spatial], [N_batch, N2, D_spatial]
    #   scores1, scores2: [N_batch, N1, 1], [N_batch, N2, 1]
    #   lang_feat: [N_batch, D_lang]
    # Output shape:
    #   relationship_scores: [N_batch, N1, N2, 1]
    #
    # This function is not responsible for initializing the variables. Please
    # handle variable initialization outside.

    with tf.variable_scope(scope, reuse=reuse):
        # An embedding module that maps the visual feature plus the spatial feature
        # linearly to the same dimension as the language feature
        N_batch = tf.shape(lang_feat)[0]
        D_lang = lang_feat.get_shape().as_list()[-1]

        N1 = tf.shape(spatial_feat1)[1]
        N2 = tf.shape(spatial_feat2)[1]

        D_spatial = spatial_feat1.get_shape().as_list()[-1]

        # Tiled spatial features of size [N_batch, N1, N2, 5*2], such that
        # spatial_feat_tiled[k, i, j] = [ spatial_feat1[k, i], spatial_feat1[k, j] ]
        spatial_feat_tiled = tf.reshape(tf.concat([
            tf.tile(tf.reshape(spatial_feat1, to_T([N_batch, -1, 1, D_spatial])),
                    to_T([1, 1, N2, 1])),
            tf.tile(tf.reshape(spatial_feat2, to_T([N_batch, 1, -1, D_spatial])),
                    to_T([1, N1, 1, 1]))
        ], axis=2), [-1, D_spatial*2])

        # Embedded spatial feature of size [N_batchxN1xN2, D_lang]
        spatial_embed = fc('spatial_embed', spatial_feat_tiled, output_dim=D_lang)

        # Elementwise multiplication with language feature and l2-normalization
        spatial_embed = tf.reshape(spatial_embed, to_T([N_batch, -1, D_lang]))
        lang_feat = tf.reshape(lang_feat, [-1, 1, D_lang])
        eltwise_mult = tf.nn.l2_normalize(spatial_embed * lang_feat, 2)
        # eltwise_mult has shape [N_batchxN1xN2, D_lang]
        eltwise_mult = tf.reshape(eltwise_mult, [-1, D_lang])

        # Localization scores as linear classification over the l2-normalized
        relationship_scores = fc('relationship_scores', eltwise_mult, output_dim=1)
        relationship_scores = tf.reshape(relationship_scores, to_T([N_batch, N1, N2, 1]))
        # Rescale the scores, if specified
        if rescale_scores:
            alpha_obj1 = tf.get_variable("alpha_obj1", shape=[], dtype=tf.float32,
                                         initializer=tf.constant_initializer(1))
            alpha_obj2 = tf.get_variable("alpha_obj2", shape=[], dtype=tf.float32,
                                         initializer=tf.constant_initializer(1))
            alpha_rel = tf.get_variable("alpha_rel", shape=[], dtype=tf.float32,
                                         initializer=tf.constant_initializer(1))
            scores1 = tf.multiply(scores1, alpha_obj1)
            scores2 = tf.multiply(scores2, alpha_obj2)
            relationship_scores = tf.multiply(relationship_scores, alpha_rel)

        final_scores = tf.add(tf.add(tf.reshape(scores1, to_T([N_batch, N1, 1, 1])),
                                     tf.reshape(scores2, to_T([N_batch, 1, N2, 1]))),
                              relationship_scores)
        final_scores.set_shape([None, None, None, 1])

    return final_scores
