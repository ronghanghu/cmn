from models import modules, fastrcnn_vgg_net, lstm_net

import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

def visgeno_baseline_net(input_batch, bbox_batch, spatial_batch, expr_obj,
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

    # Apply the same LSTM network on all expressions to extract their language
    # features.
    lang_obj = lstm_net.lstm_encoder(expr_obj, "lstm", num_vocab=num_vocab,
        embed_dim=embed_dim, lstm_dim=lstm_dim, apply_dropout=lstm_dropout)

    scores_obj = modules.localization_module_grid_score(vis_feat, spatial_batch, lang_obj)

    return scores_obj
