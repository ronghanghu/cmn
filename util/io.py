# -*- coding: utf-8 -*-

import json
import numpy as np
import scipy.io as sio

def load_str_list(filename):
    with open(filename, 'r') as f:
        str_list = f.readlines()
    str_list = [s[:-1] for s in str_list]
    return str_list

def save_str_list(str_list, filename):
    str_list = [s+'\n' for s in str_list]
    with open(filename, 'w') as f:
        f.writelines(str_list)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(json_obj, filename):
    with open(filename, 'w') as f:
        json.dump(json_obj, f, separators=(',\n', ':\n'), cls=MyEncoder)

def load_numpy_obj(filename):
    return np.load(filename)[()]

def save_numpy_obj(obj, filename):
    return np.save(filename, np.array(obj, dtype=np.object))

def load_refclef_gt_mask(mask_path):
    mat = sio.loadmat(mask_path)
    mask = (mat['segimg_t'] == 0)
    return mask

def load_proposal_mask(mask_path):
    mat = sio.loadmat(mask_path)
    mask = mat['mask']
    return mask.transpose((2, 0, 1))
