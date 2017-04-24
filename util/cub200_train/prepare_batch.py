from __future__ import absolute_import, division, print_function

import skimage.io
import skimage.transform
import numpy as np

from models.spatial_feat import spatial_feature_from_bbox
from util import im_processing, text_processing
from util.eval_tools import compute_bboxes_iou_mat as grid_ious

def prepare_one_image_roi(im, im_mean, proposal_bboxes, min_size, max_size):
    # calculate the resize scaling factor
    im_h, im_w = im.shape[:2]
    # make the short size equal to min_size but also the long size no bigger than max_size
    scale = min(max(min_size/im_h, min_size/im_w), max_size/im_h, max_size/im_w)

    # resize and process the image
    new_h, new_w = int(scale*im_h), int(scale*im_w)
    im_resized = skimage.img_as_float(skimage.transform.resize(im, [new_h, new_w]))
    im_processed = im_resized*255 - im_mean
    im_batch = im_processed[np.newaxis, ...].astype(np.float32)

    proposal_bboxes = proposal_bboxes * scale
    proposal_bboxes = im_processing.rectify_bboxes(proposal_bboxes, height=new_h, width=new_w)

    # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    bbox_batch = np.zeros((len(proposal_bboxes), 5), np.float32)
    bbox_batch[:, 1:5] = proposal_bboxes
    spatial_batch = spatial_feature_from_bbox(proposal_bboxes, im_h=new_h, im_w=new_w)

    return im_batch, bbox_batch, spatial_batch

def load_one_batch(iminfo, im_mean, min_size, max_size, proposal_name,
    vocab_dict, T, iou_thresh, include_gt_bbox, softmax_label):

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
    region_bboxes *= scale
    region_bboxes = im_processing.rectify_bboxes(region_bboxes, height=new_h, width=new_w)

    # language sequences
    text_seq_batch = np.zeros((T, len(regions)), np.int32)
    for n in range(len(regions)):
        text_seq_batch[:, n] = text_processing.preprocess_sentence(regions[n][1], vocab_dict, T)

    # resize the region proposals and compute spatial features
    if proposal_name is not None:
        proposal_bboxes = np.array(iminfo[proposal_name], np.float32)
        proposal_bboxes *= scale
        proposal_bboxes = im_processing.rectify_bboxes(proposal_bboxes, height=new_h, width=new_w)
        # add ground-truth regions to the proposals if specified so
        if include_gt_bbox:
            proposal_bboxes = np.concatenate((region_bboxes, proposal_bboxes))
    else:
        assert(include_gt_bbox)
        proposal_bboxes = region_bboxes

    # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    bbox_batch = np.zeros((len(proposal_bboxes), 5), np.float32)
    bbox_batch[:, 1:5] = proposal_bboxes
    spatial_batch = spatial_feature_from_bbox(proposal_bboxes, im_h=new_h, im_w=new_w)

    # labels
    iou_mat = grid_ious(region_bboxes, proposal_bboxes)
    if softmax_label:
        labels = np.argmax(iou_mat, axis=1)
        label_batch = labels.astype(np.int32)
    else:
        # put query number at the first dimension
        # the output will be [N_lan, N_vis, 1] format
        labels = (iou_mat >= iou_thresh)
        
#         # Merge same sentences and the corresponding labels
#         # if a proposal bbox is True with one "person" region and False with
#         # another "person" region, then this proposal should be "True" with
#         # the unique sentence "person" after merge.
#         keep = np.ones(len(regions), dtype=np.bool)
#         sentence_dict = {}
#         for n in range(len(regions)):
#             s = regions[n][1]
#             if s not in sentence_dict:
#                 keep[n] = True
#                 sentence_dict[s] = n
#             else:
#                 # Merge this sentence with previous sentence at n_merge
#                 keep[n] = False
#                 n_merge = sentence_dict[s]
#                 labels[n_merge, :] = np.logical_or(labels[n_merge, :], labels[n, :])

        label_batch = labels.astype(np.float32)
#         if not np.all(keep):
#             # Keep only unique sentences after merging
#             text_seq_batch = text_seq_batch[:, keep]
#             label_batch = label_batch[keep, :]

    batch=dict(text_seq_batch=text_seq_batch, im_batch=im_batch,
               bbox_batch=bbox_batch, spatial_batch=spatial_batch,
               label_batch=label_batch)

    return batch
