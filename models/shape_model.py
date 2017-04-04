from models import modules, vgg_net, lstm_net

import tensorflow as tf

def shape_attention_net(input_batch, spatial_batch, expr_obj, num_vocab,
    embed_dim, lstm_dim, vgg_dropout, lstm_dropout,
    share_localization_params=True, rescale_scores=True):
    #   bbox_batch has shape [N_box, 5] and
    #   spatial_batch has shape [N_box, D_spatial] and
    #   expr_obj has shape [T, N_batch]

    N_batch = tf.shape(expr_obj)[1]
    N_box = tf.shape(spatial_batch)[0]

    # Extract visual features
    vis_feat = vgg_net.vgg_fc7(input_batch, "vgg_local", apply_dropout=vgg_dropout)
    D_vis = vis_feat.get_shape().as_list()[-1]

    # Extract BoW representation using attention
    lang_obj1, lang_obj2, lang_relation = lstm_net.attbilstm_simple(
        expr_obj, "lstm", num_vocab=num_vocab, embed_dim=embed_dim,
        lstm_dim=lstm_dim, apply_dropout=lstm_dropout)

    if share_localization_params:
        # Score for each bounding box matching the first object
        # scores_obj1 has shape [N_batch, N_box, 1]
        scores_obj1 = modules.localization_module_grid_score(vis_feat,
            spatial_batch, lang_obj1)
        # Score for each bounding box matching the second object
        # scores_obj2 has shape [N_batch, N_box, 1]
        scores_obj2 = modules.localization_module_grid_score(vis_feat,
            spatial_batch, lang_obj2, reuse=True)
    else:
        # Score for each bounding box matching the first object
        # scores_obj1 has shape [N_batch, N_box, 1]
        scores_obj1 = modules.localization_module_grid_score(vis_feat,
            spatial_batch, lang_obj1)
        # Score for each bounding box matching the second object
        # scores_obj2 has shape [N_batch, N_box, 1]
        scores_obj2 = modules.localization_module_grid_score(vis_feat,
            spatial_batch, lang_obj2, scope="localization_module2")

    # Scores for each pair of bounding box matching the relationship
    # Tile the scores by broadcasting add
    # scores_rel has shape [N_batch, N_box, N_box, 1]
    scores_rel = modules.relationship_module_spatial_only_grid_score(
        spatial_batch, scores_obj1, spatial_batch, scores_obj2, lang_relation,
        rescale_scores=rescale_scores)
    tf.add_to_collection("shape_attention_net/scores_rel", scores_rel)

    return scores_rel
