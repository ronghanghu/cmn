from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

def print_bbox(bboxes, style='r-'):
    """A utility function to help visualizing boxes."""
    bboxes = np.array(bboxes).reshape((-1, 4))
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        plt.plot([xmin, xmax, xmax, xmin, xmin],
                 [ymin, ymin, ymax, ymax, ymin], style)
