from models import modules, fastrcnn_vgg_net, lstm_net

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

def visual7w_attbilstm_net(input_batch, bbox_batch1, spatial_batch1,
    bbox_batch2, spatial_batch2, expr_obj, num_vocab, embed_dim, lstm_dim,
    vgg_dropout, lstm_dropout):
    # a sentence is parsed into [expr_obj1, expr_relation, expr_obj2]
    #   bbox_batch1 has shape [N_batch*N1, 5] and
    #   spatial_batch1 has shape [N_batch, N1, D_spatial] and
    #   bbox_batch2 has shape [N2, 5] and
    #   spatial_batch2 has shape [1, N2, D_spatial] and
    #   expr_obj has shape [T, N_batch]
    # where N1 is the number of choices (= 4 in Visual 7W) and
    # N2 is the number of proposals (~ 300 for RPN in Faster RCNN)

    N_batch = tf.shape(spatial_batch1)[0]
    N1 = tf.shape(spatial_batch1)[1]
    N2 = tf.shape(spatial_batch2)[1]

    # Extract visual features
    vis_feat1 = fastrcnn_vgg_net.vgg_roi_fc7(input_batch,
        tf.reshape(bbox_batch1, [-1, 5]), "vgg_local",
        apply_dropout=vgg_dropout)
    D_vis = vis_feat1.get_shape().as_list()[-1]
    vis_feat1 = tf.reshape(vis_feat1, to_T([N_batch, N1, D_vis]))
    vis_feat1.set_shape([None, None, D_vis])

    # Reshape and tile vis_feat2 and spatial_batch2
    vis_feat2 = fastrcnn_vgg_net.vgg_roi_fc7(input_batch,
        tf.reshape(bbox_batch2, [-1, 5]), "vgg_local",
        apply_dropout=vgg_dropout, reuse=True)
    vis_feat2 = tf.reshape(vis_feat2, to_T([1, N2, D_vis]))
    vis_feat2 = tf.tile(vis_feat2, to_T([N_batch, 1, 1]))
    vis_feat2.set_shape([None, None, D_vis])
    spatial_batch2 = tf.tile(spatial_batch2, to_T([N_batch, 1, 1]))

    # Extract representation using attention
    lang_obj1, lang_obj2, lang_relation = lstm_net.attbilstm(
        expr_obj, "lstm", num_vocab=num_vocab, embed_dim=embed_dim,
        lstm_dim=lstm_dim, apply_dropout=lstm_dropout)

    # Score for each bounding box matching the first object
    # scores_obj1 has shape [N_batch, N1, 1]
    scores_obj1 = modules.localization_module_batch_score(vis_feat1,
        spatial_batch1, lang_obj1)
    # Score for each bounding box matching the second object
    # scores_obj2 has shape [N_batch, N2, 1]
    scores_obj2 = modules.localization_module_batch_score(vis_feat2,
        spatial_batch2, lang_obj2, reuse=True)

    # Scores for each pair of bounding box matching the relationship
    # Tile the scores by broadcasting add
    # scores_rel has shape [N_batch, N1, N2, 1]
    scores_rel = modules.relationship_module_spatial_only_batch_score(
        spatial_batch1, scores_obj1, spatial_batch2, scores_obj2, lang_relation,
        rescale_scores=True)
    # marginal_scores has shape [N_batch, N1, 1]
    tf.add_to_collection("s_pair", scores_rel)

    marginal_scores = tf.reduce_max(scores_rel, reduction_indices=2)
    final_scores = tf.reshape(marginal_scores, to_T([N_batch, -1]))

    return final_scores
