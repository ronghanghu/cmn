from __future__ import absolute_import, division, print_function

import os
import sys
import threading
import queue
import numpy as np

import util.io
from util import text_processing
from util.visgeno_rel_train.prepare_batch import load_one_batch

def run_prefetch(prefetch_queue, imdb, im_mean, min_size, max_size, vocab_dict,
    T, num_batch, shuffle, max_bbox_num, max_rel_num):

    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]

        try:
            batch = load_one_batch(imdb[batch_id], im_mean, min_size, max_size,
                                   vocab_dict, T, max_bbox_num, max_rel_num)
            # add loaded batch to fetchqing queue
            prefetch_queue.put(batch, block=True)
        except IOError as err:
            # Print error and move on to next batch
            print('data reader: skipped an image.', err)
            prefetch_queue.put(None, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, imdb_file, vocab_file, im_mean, min_size=600,
        max_size=1000, T=20, shuffle=True, prefetch_num=8,
        max_bbox_num=80, max_rel_num=20):

        print('Loading ROI data from file...', end=''); sys.stdout.flush()
        if imdb_file.endswith('.json'):
            imdb = util.io.load_json(imdb_file)
        elif imdb_file.endswith('.npy'):
            imdb = util.io.load_numpy_obj(imdb_file)
        else:
            raise TypeError('unknown imdb format.')
        self.imdb = imdb
        print('Done.')

        self.vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
        self.im_mean = im_mean

        self.min_size = min_size
        self.max_size = max_size

        self.T = T

        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        self.num_batch = len(self.imdb)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.imdb, self.im_mean, self.min_size,
                  self.max_size, self.vocab_dict, self.T, self.num_batch,
                  self.shuffle, max_bbox_num, max_rel_num))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self):
        print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch + 1, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch
