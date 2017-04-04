from __future__ import absolute_import, division, print_function

import skimage.io
import skimage.transform
import numpy as np

from models.spatial_feat import spatial_feature_from_bbox
from util import im_processing, text_processing

def load_one_batch(iminfo, im_mean, min_size, max_size, vocab_dict, T,
        max_bbox_num, max_rel_num):
    im_path = iminfo['im_path']
    im = skimage.io.imread(im_path)
    if im.ndim == 2:
        im = np.tile(im[..., np.newaxis], (1, 1, 3))

    # calculate the resize scaling factor
    im_h, im_w = im.shape[:2]
    # make the short size equal to min_size but also the long size no bigger than max_size
    scale = min(max(min_size/im_h, min_size/im_w), max_size/im_h, max_size/im_w)

    # resize and process the image
    new_h, new_w = int(scale*im_h), int(scale*im_w)
    im_resized = skimage.img_as_float(skimage.transform.resize(im, [new_h, new_w]))
    im_processed = im_resized*255 - im_mean
    im_batch = im_processed[np.newaxis, ...].astype(np.float32)

    # annotate regions
    bboxes = np.array(iminfo['bboxes'], np.float32)
    bboxes = bboxes[:max_bbox_num]
    if len(bboxes) == 0:
        raise IOError('no object annotations for image ' + im_path)
    bboxes *= scale
    bboxes = im_processing.rectify_bboxes(bboxes, height=new_h, width=new_w)
    spatial_batch = spatial_feature_from_bbox(bboxes, im_h=new_h, im_w=new_w)

    # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    bbox_batch = np.zeros((len(bboxes), 5), np.float32)
    bbox_batch[:, 1:5] = bboxes

    # Randomly pick one relationship from all the relationships in this image
    mapped_rels = iminfo['mapped_rels']
    if len(mapped_rels) == 0:
        raise IOError('no relationship annotations for image ' + im_path)
    # Prune the relationships to avoid objects out of max_bbox_num
    if len(iminfo['bboxes']) > max_bbox_num:
        mapped_rels = [rel for rel in mapped_rels if max(rel[0], rel[1]) < max_bbox_num]
    if len(mapped_rels) > max_rel_num:
        mapped_rels = [mapped_rels[n] for n in np.random.choice(len(mapped_rels), max_rel_num)]
    num_rels = len(mapped_rels)
    if num_rels == 0:
        raise IOError('no relationship annotations for image ' + im_path)

    expr_obj1_batch = np.zeros((T, num_rels), dtype=np.int32)
    expr_obj2_batch = np.zeros((T, num_rels), dtype=np.int32)
    expr_relation_batch = np.zeros((T, num_rels), dtype=np.int32)
    text_seq_batch = np.zeros((T, num_rels), dtype=np.int32)
    label_batch = np.zeros(num_rels, dtype=np.int32)
    label_weak_batch = np.zeros(num_rels, dtype=np.int32)
    label_weak_obj2_batch = np.zeros(num_rels, dtype=np.int32)
    questions = [None for _ in range(num_rels)]
    obj1_component_idx = np.zeros((T, num_rels), np.bool)
    obj2_component_idx = np.zeros((T, num_rels), np.bool)
    rel_component_idx = np.zeros((T, num_rels), np.bool)
    for n_rel in range(num_rels):
        obj1_idx, obj2_idx, obj1_name, predcate_name, obj2_name = mapped_rels[n_rel]
        question = obj1_name + ' ' + predcate_name + ' ' + obj2_name

        vocabidx_obj1 = text_processing.sentence2vocab_indices(obj1_name, vocab_dict)
        vocabidx_obj2 = text_processing.sentence2vocab_indices(obj2_name, vocab_dict)
        vocabidx_predcate = text_processing.sentence2vocab_indices(predcate_name, vocab_dict)

        expr_obj1_batch[:, n_rel] = text_processing.preprocess_vocab_indices(vocabidx_obj1, vocab_dict, T)
        expr_obj2_batch[:, n_rel] = text_processing.preprocess_vocab_indices(vocabidx_obj2, vocab_dict, T)
        expr_relation_batch[:, n_rel] = text_processing.preprocess_vocab_indices(vocabidx_predcate, vocab_dict, T)
        text_seq_batch[:, n_rel] = text_processing.preprocess_vocab_indices(
            vocabidx_obj1 + vocabidx_predcate + vocabidx_obj2, vocab_dict, T)

        l_obj1, l_obj2, l_rel = len(vocabidx_obj1), len(vocabidx_obj2), len(vocabidx_predcate)
        obj1_component_idx[-l_obj1-l_rel-l_obj2:-l_rel-l_obj2, n_rel] = True
        rel_component_idx[-l_rel-l_obj2:-l_obj2, n_rel] = True
        obj2_component_idx[-l_obj2:, n_rel] = True

        label_batch[n_rel] = obj1_idx*bbox_batch.shape[0] + obj2_idx
        label_weak_batch[n_rel] = obj1_idx
        label_weak_obj2_batch[n_rel] = obj2_idx
        questions[n_rel] = question

    batch=dict(im_batch=im_batch, bbox_batch=bbox_batch, spatial_batch=spatial_batch,
               expr_obj1_batch=expr_obj1_batch, expr_obj2_batch=expr_obj2_batch,
               expr_relation_batch=expr_relation_batch,
               text_seq_batch=text_seq_batch, label_weak_batch=label_weak_batch,
               label_weak_obj2_batch=label_weak_obj2_batch,
               label_batch=label_batch, questions=questions,
               obj1_component_idx=obj1_component_idx,
               obj2_component_idx=obj2_component_idx,
               rel_component_idx=rel_component_idx)

    return batch
