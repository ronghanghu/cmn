from __future__ import absolute_import, division, print_function

import skimage.io
import skimage.transform
import numpy as np

from models.spatial_feat import spatial_feature_from_bbox
from util import im_processing, text_processing
from util.eval_tools import compute_bboxes_iou_mat as grid_ious

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

    # annotate regions
    regions = iminfo['regions']
    if len(regions) == 0:
        raise IOError('no region annotations for image ' + im_path)
    region_bboxes = np.array([ann[0] for ann in regions], np.float32)
    # save coco_bboxes, needed for evaluation code
    coco_bboxes = region_bboxes.copy()
    # back to [x, y, w, h]
    coco_bboxes[:, 2:4] = coco_bboxes[:, 2:4] - coco_bboxes[:, 0:2] + 1
    region_bboxes *= scale
    region_bboxes = im_processing.rectify_bboxes(region_bboxes, height=new_h, width=new_w)

    # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    bbox_batch = np.zeros((len(region_bboxes), 5), np.float32)
    bbox_batch[:, 1:5] = region_bboxes
    spatial_batch = spatial_feature_from_bbox(region_bboxes, im_h=new_h, im_w=new_w)

    # a region may have zero, one or more sentence annotations
    # align language sequences with regions
    text_seq_batch = []
    label_batch = []
    coco_ann_ids = []  # needed for evaluation code
    questions = []  # needed for evaluation code
    for n in range(len(regions)):
        for n_s in range(len(regions[n][1])):
            s = regions[n][1][n_s]
            text_seq_batch.append(text_processing.preprocess_sentence(s, vocab_dict, T))
            label_batch.append(n)
            coco_ann_ids.append(regions[n][2])
            questions.append(s)

    text_seq_batch = np.array(text_seq_batch, dtype=np.int32).T

    label_batch = np.array(label_batch, dtype=np.int32)

    batch=dict(text_seq_batch=text_seq_batch, im_batch=im_batch,
               bbox_batch=bbox_batch, spatial_batch=spatial_batch,
               label_batch=label_batch, coco_ann_ids=coco_ann_ids,
               questions=questions, coco_bboxes=coco_bboxes)

    return batch
