import numpy as np
import os.path
import sys
import warnings

_project_folder_ = os.path.abspath('../')
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from scripts.base_train import SketchR2CNNTrain


class TUBerlinSketchR2CNNTrain(SketchR2CNNTrain):

    def __init__(self):
        super().__init__()
        self.chosen_fold = 0

    def run_name(self):
        return 'fold{}'.format(self.chosen_fold)

    def set_fold(self, idx):
        self.chosen_fold = idx

    def prepare_dataset(self, dataset_dict):
        for m in dataset_dict:
            dataset_dict[m].set_fold(self.chosen_fold)

    def checkpoint_prefix(self):
        ckpt = self.config['ckpt_prefix']
        return ckpt.format(self.chosen_fold)


if __name__ == '__main__':
    accus = list()
    with TUBerlinSketchR2CNNTrain() as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for idx in range(3):
                app.set_fold(idx)
                accu = app.run()
                accus.append(accu)
    print('Cross validation: {}; mean = {}'.format(accus, np.mean(accus)))
