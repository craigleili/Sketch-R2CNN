import os.path
import pickle
import sys
import warnings

_project_folder_ = os.path.abspath('../')
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from scripts.base_eval import SketchR2CNNEval

if __name__ == '__main__':
    app = SketchR2CNNEval()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        accuracies, stats = app.run()

        print('Progressive Recognition Accuracies:\n')
        print(accuracies)
