from __future__ import division
from __future__ import print_function

from .sketch_util import SketchUtil
from torch.utils.data import Dataset
import numpy as np
import pickle
import random


class TUBerlinDataset(Dataset):

    def __init__(self, pkl_file, mode, drop_strokes=True):
        self.pkl_file = pkl_file
        self.mode = mode
        self.drop_strokes = drop_strokes

        with open(self.pkl_file, 'rb') as fh:
            saved = pickle.load(fh)
            self.categories = saved['categories']
            self.sketches = saved['sketches']
            self.cvxhulls = saved['convex_hulls']
            self.folds = saved['folds']

        self.fold_idx = None
        self.indices = list()

    def set_fold(self, idx):
        self.fold_idx = idx
        self.indices = list()

        if self.mode == 'train':
            for i in range(len(self.folds)):
                if i != idx:
                    self.indices.extend(self.folds[i])
        else:
            self.indices = self.folds[idx]

        print('[*] Created a new {} dataset with {} fold as validation data'.format(self.mode, idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cid, sid = self.indices[idx]

        sid_points = np.copy(self.sketches[cid][sid])

        if self.mode == 'train':
            cvxhull = self.cvxhulls[cid][sid]
            pts_xy = sid_points[:, 0:2]
            if cvxhull is not None:
                if random.uniform(0, 1) > 0.5:
                    pts_xy = SketchUtil.random_cage_deform(np.copy(cvxhull), pts_xy, thresh=0.1)
                    pts_xy = SketchUtil.normalization(pts_xy)
                if random.uniform(0, 1) > 0.5:
                    pts_xy = SketchUtil.random_affine_transform(pts_xy, scale_factor=0.2, rot_thresh=40.0)
            pts_xy = SketchUtil.random_horizontal_flip(pts_xy)
            sid_points[:, 0:2] = pts_xy
            if self.drop_strokes:
                sid_points = self._random_drop_strokes(sid_points)
        sample = {'points3': sid_points, 'category': cid}
        return sample

    def _random_drop_strokes(self, points3):
        strokes = SketchUtil.to_stroke_list(points3)
        num_strokes = len(strokes)
        if num_strokes < 2:
            return points3
        sort_idxes = SketchUtil.compute_stroke_orders([s[:, 0:2] for s in strokes])
        keep_prob = np.random.uniform(0, 1, num_strokes)
        keep_prob[:(num_strokes // 2)] = 1
        keep_idxes = np.array(sort_idxes, np.int32)[keep_prob > 0.5]
        keep_strokes = [strokes[i] for i in sorted(keep_idxes.tolist())]
        return np.concatenate(keep_strokes, axis=0)

    def num_categories(self):
        return len(self.categories)

    def dispose(self):
        pass

    def get_name_prefix(self):
        return 'TUBerlin-{}-{}'.format(self.mode, self.fold_idx)
