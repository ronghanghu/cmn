from __future__ import absolute_import, division, print_function

import numpy as np
#import pyximport; pyximport.install()
#from util.nms import cpu_nms as nms

def compute_accuracy(scores, labels):
    is_pos = (labels != 0)
    is_neg = np.logical_not(is_pos)
    num_pos = np.sum(is_pos)
    num_neg = np.sum(is_neg)
    num_all = num_pos + num_neg

    is_correct = np.logical_xor(scores < 0, is_pos)
    accuracy_all = np.sum(is_correct) / num_all
    accuracy_pos = np.sum(is_correct[is_pos]) / num_pos
    accuracy_neg = np.sum(is_correct[is_neg]) / num_neg
    accuracy_all = 0 if np.isnan(accuracy_all) else accuracy_all
    accuracy_pos = 0 if np.isnan(accuracy_pos) else accuracy_pos
    accuracy_neg = 0 if np.isnan(accuracy_neg) else accuracy_neg
    return accuracy_all, accuracy_pos, accuracy_neg

# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_bbox_iou(bboxes, target):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    if isinstance(target, list):
        target = np.array(target)
    target = target.reshape((-1, 4))

    A_bboxes = (bboxes[..., 2]-bboxes[..., 0]+1) * (bboxes[..., 3]-bboxes[..., 1]+1)
    A_target = (target[..., 2]-target[..., 0]+1) * (target[..., 3]-target[..., 1]+1)
    assert(np.all(A_bboxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(bboxes[..., 0], target[..., 0])
    I_y1 = np.maximum(bboxes[..., 1], target[..., 1])
    I_x2 = np.minimum(bboxes[..., 2], target[..., 2])
    I_y2 = np.minimum(bboxes[..., 3], target[..., 3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_bboxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

def compute_bboxes_iou_mat(bboxes1, bboxes2):
    """
    bboxes1 is N1 x 4, bboxes 2 is N2 x 4, both [xmin, ymin, xmax, ymax] format,
      0-indexed, including xmax and ymax

    Returns a N1 x N2 matrix of IoU between boxes in bboxes1 and bboxes2
    """

    if isinstance(bboxes1, list):
        bboxes1 = np.array(bboxes1)
    bboxes1 = bboxes1.reshape((-1, 1, 4))

    if isinstance(bboxes2, list):
        bboxes2 = np.array(bboxes2)
    bboxes2 = bboxes2.reshape((1, -1, 4))

    A_bboxes1 = (bboxes1[..., 2]-bboxes1[..., 0]+1) * (bboxes1[..., 3]-bboxes1[..., 1]+1)
    A_bboxes2 = (bboxes2[..., 2]-bboxes2[..., 0]+1) * (bboxes2[..., 3]-bboxes2[..., 1]+1)
    assert(np.all(A_bboxes1 >= 0))
    assert(np.all(A_bboxes2 >= 0))
    I_x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    I_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    I_x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    I_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_bboxes1 + A_bboxes2 - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

# # all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U
