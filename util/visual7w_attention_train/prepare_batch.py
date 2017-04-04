from __future__ import absolute_import, division, print_function

import skimage.io
import skimage.transform
import numpy as np

from models.spatial_feat import spatial_feature_from_bbox
from util import im_processing, text_processing

def load_one_batch(iminfo, im_mean, min_size, max_size, vocab_dict, T):
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

    # Sample one qa pair from all QA pairs
    qa_pairs = iminfo['processed_qa_pairs']

    num_questions = len(qa_pairs)
    num_choices = 4
    text_seq_batch = np.zeros((T, num_questions), np.int32)
    label_batch = np.zeros(num_questions, np.int32)
    bboxes = np.zeros((num_questions, num_choices, 4), np.float32)
    questions = [None for _ in range(num_questions)]
    for n_q in range(num_questions):
        this_bboxes, question, label, relationship = qa_pairs[n_q]
        bboxes[n_q, :, :] = this_bboxes
        text_seq_batch[:, n_q] = text_processing.preprocess_sentence(question, vocab_dict, T)
        label_batch[n_q] = label
        questions[n_q] = question

    # annotate regions
    bboxes = bboxes.reshape((-1, 4)) * scale
    bboxes = im_processing.rectify_bboxes(bboxes, height=new_h, width=new_w)
    spatial_batch1 = spatial_feature_from_bbox(bboxes, im_h=new_h, im_w=new_w)
    spatial_batch1 = spatial_batch1.reshape((num_questions, num_choices, 5))
    # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    bbox_batch1 = np.zeros((len(bboxes), 5), np.float32)
    bbox_batch1[:, 1:5] = bboxes

    num_proposals = len(iminfo['proposals'])
    proposals = iminfo['proposals']
    proposals = proposals * scale
    proposals = im_processing.rectify_bboxes(proposals, height=new_h, width=new_w)
    spatial_batch2 = spatial_feature_from_bbox(proposals, im_h=new_h, im_w=new_w)
    spatial_batch2 = spatial_batch2.reshape((1, num_proposals, 5))
    # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    bbox_batch2 = np.zeros((len(proposals), 5), np.float32)
    bbox_batch2[:, 1:5] = proposals

    batch=dict(questions=questions, im_batch=im_batch,
               bbox_batch1=bbox_batch1, spatial_batch1=spatial_batch1,
               bbox_batch2=bbox_batch2, spatial_batch2=spatial_batch2,
               text_seq_batch=text_seq_batch, label_batch=label_batch)

    return batch
