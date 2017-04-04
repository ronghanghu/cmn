from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

import tensorflow.python.kernel_tests.gradient_checker as gradient_checker
import roi_pooling_op_grad

roi_pool = roi_pooling_op_grad.module.roi_pool

# Check gradient for ROI Pooling
data = tf.random_normal([2, 10, 12, 3])
rois = tf.constant([[0, 1, 2, 4, 8], [0, 2, 2, 4, 6],
                    [1, 2, 3, 9, 8], [1, 4, 5, 8, 9]], dtype=tf.float32)
pool, _ = roi_pool(data, rois, pooled_height=7, pooled_width=7,
                   spatial_scale=1./16)

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
with tf.device("/gpu:0"):
    error = gradient_checker.compute_gradient_error(data, [2, 10, 12, 3],
                                                    pool, [4, 7, 7, 3])
sess.close()

if error < 1e-4:
    print("Gradient check passed.")
else:
    print("Gradient check failed. Maximum gradient error: %f" % error)
