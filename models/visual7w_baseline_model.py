from models import modules, fastrcnn_vgg_net, lstm_net

import tensorflow as tf

def visual7w_baseline_net(input_batch, bbox_batch, spatial_batch, expr_obj, num_vocab,
                          embed_dim, lstm_dim, vgg_dropout, lstm_dropout):
    # Extract visual features
    vis_feat = fastrcnn_vgg_net.vgg_roi_fc7(input_batch, bbox_batch,
        "vgg_local", apply_dropout=vgg_dropout)

    # Apply LSTM network to extract language features.
    lang_obj = lstm_net.lstm_encoder(expr_obj, "lstm", num_vocab=num_vocab,
        embed_dim=embed_dim, lstm_dim=lstm_dim, apply_dropout=lstm_dropout)

    # Score for each bounding box matching the object
    scores_obj = modules.localization_module(vis_feat, spatial_batch, lang_obj)
    return scores_obj
