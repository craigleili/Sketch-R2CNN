import numpy as np
import os.path
import sys
import warnings

_project_folder_ = os.path.abspath('../')
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from scripts.base_eval import SketchR2CNNEval


class TUBerlinSketchR2CNNEval(SketchR2CNNEval):

    def __init__(self):
        super().__init__()
        self.chosen_fold = 0

    def set_fold(self, idx):
        self.chosen_fold = idx

    def prepare_dataset(self, dataset):
        super().prepare_dataset(dataset)
        dataset.set_fold(self.chosen_fold)

    def checkpoint_prefix(self):
        ckpt = self.config['checkpoint']
        return ckpt.format(self.chosen_fold)


if __name__ == '__main__':
    app = TUBerlinSketchR2CNNEval()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        xvalid_accus = list()
        for fidx in range(3):
            app.set_fold(fidx)
            accuracies, stats = app.run()
            xvalid_accus.append(accuracies)
        avg_xvalid_accus = np.mean(np.array(xvalid_accus, dtype=np.float32), axis=0)
        print('Progressive Recognition Accuracies:\n')
        print(avg_xvalid_accus)
