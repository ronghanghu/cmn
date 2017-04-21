from __future__ import absolute_import, division, print_function

import os
import sys
import threading
import queue
import numpy as np

import util.io
from util import text_processing
from util.fastrcnn_train.prepare_batch import load_one_batch

def run_prefetch(prefetch_queue, roidb, im_mean, min_size, max_size,
    proposal_name, vocab_dict, T, iou_thresh, include_gt_bbox, softmax_label,
    num_batch, shuffle):

    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]
        batch = load_one_batch(roidb[batch_id], im_mean, min_size, max_size,
                               proposal_name, vocab_dict, T, iou_thresh,
                               include_gt_bbox, softmax_label)

        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, roidb_file, vocab_file, im_mean, proposal_name,
        include_gt_bbox, softmax_label, min_size=600, max_size=1000, T=20,
        iou_thresh=.5, shuffle=True, prefetch_num=8):

        print('Loading ROI data from file...', end=''); sys.stdout.flush()
        if isinstance(roidb_file, list):
            roidb = []
            for fname in roidb_file:
                roidb += util.io.load_json(fname)
        else:
            roidb = util.io.load_json(roidb_file)
        self.roidb = roidb
        print('Done.')

        self.vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
        self.im_mean = im_mean
        self.proposal_name = proposal_name
        self.include_gt_bbox = include_gt_bbox
        self.softmax_label = softmax_label

        self.min_size = min_size
        self.max_size = max_size

        self.T = T
        self.iou_thresh = iou_thresh

        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        self.num_batch = len(self.roidb)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.roidb, self.im_mean, self.min_size,
                  self.max_size, self.proposal_name, self.vocab_dict, self.T,
                  self.iou_thresh, self.include_gt_bbox, self.softmax_label,
                  self.num_batch, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self):
        print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch
