from __future__ import absolute_import, division, print_function

import numpy as np

def spatial_feature_from_bbox(bboxes, im_h, im_w):
    # Generate 5-dimensional spatial features from the image
    # [xmin, ymin, xmax, ymax, S] where S is the area of the box
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))
    # Check the size of the bounding boxes
    assert(np.all(bboxes[:, 0:2] >= 0))
    assert(np.all(bboxes[:, 0] <= bboxes[:, 2]))
    assert(np.all(bboxes[:, 1] <= bboxes[:, 3]))
    assert(np.all(bboxes[:, 2] <= im_w))
    assert(np.all(bboxes[:, 3] <= im_h))

    feats = np.zeros((bboxes.shape[0], 5))
    feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
    feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
    feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
    feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
    feats[:, 4] = (feats[:, 2] - feats[:, 0]) * (feats[:, 3] - feats[:, 1]) # S
    return feats
