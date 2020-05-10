from __future__ import division
from __future__ import print_function

from .sketch_util import SketchUtil
from torch.utils.data import Dataset
import h5py
import numpy as np
import os.path as osp
import pickle
import random


class QuickDrawDataset(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        self.data = None

        with open(osp.join(root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']
            self.indices = saved_pkl['indices'][self.mode_indices[mode]]

        print('[*] Created a new {} dataset: {}'.format(mode, root_dir))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(osp.join(self.root_dir, 'quickdraw_{}.hdf5'.format(self.mode)), 'r')

        index_tuple = self.indices[idx]
        cid = index_tuple[0]
        sid = index_tuple[1]
        sketch_path = '/sketch/{}/{}'.format(cid, sid)

        sid_points = np.array(self.data[sketch_path][()], dtype=np.float32)
        sample = {'points3': sid_points, 'category': cid}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.data is not None:
            self.data.close()

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDraw-{}'.format(self.mode)
