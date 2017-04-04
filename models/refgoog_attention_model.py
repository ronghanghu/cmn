from models import modules, fastrcnn_vgg_net, lstm_net

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

def refgoog_attbilstm_net(input_batch, bbox_batch, spatial_batch, expr_obj,
    num_vocab, embed_dim, lstm_dim, vgg_dropout, lstm_dropout):
    #   bbox_batch has shape [N_box, 5] and
    #   spatial_batch has shape [N_box, D_spatial] and
    #   expr_obj has shape [T, N_batch]

    N_batch = tf.shape(expr_obj)[1]
    N_box = tf.shape(spatial_batch)[0]

    # Extract visual features
    vis_feat = fastrcnn_vgg_net.vgg_roi_fc7(input_batch, bbox_batch,
        "vgg_local", apply_dropout=vgg_dropout)
    D_vis = vis_feat.get_shape().as_list()[-1]

    # Extract representation using attention
    lang_obj1, lang_obj2, lang_relation = lstm_net.attbilstm(
        expr_obj, "lstm", num_vocab=num_vocab, embed_dim=embed_dim,
        lstm_dim=lstm_dim, apply_dropout=lstm_dropout)

    # Score for each bounding box matching the first object
    # scores_obj1 has shape [N_batch, N_box, 1]
    scores_obj1 = modules.localization_module_grid_score(vis_feat,
        spatial_batch, lang_obj1)
    # Score for each bounding box matching the second object
    # scores_obj2 has shape [N_batch, N_box, 1]
    scores_obj2 = modules.localization_module_grid_score(vis_feat,
        spatial_batch, lang_obj2, reuse=True)

    # Scores for each pair of bounding box matching the relationship
    # Tile the scores by broadcasting add
    # scores_rel has shape [N_batch, N_box, N_box, 1]
    scores_rel = modules.relationship_module_spatial_only_grid_score(
        spatial_batch, scores_obj1, spatial_batch, scores_obj2, lang_relation,
        rescale_scores=True)
    tf.add_to_collection("s_pair", scores_rel)

    # marginal_scores has shape [N_batch, N_box, 1]
    marginal_scores = tf.reduce_max(scores_rel, reduction_indices=2)
    final_scores = tf.reshape(marginal_scores, to_T([N_batch, -1]))

    return final_scores
