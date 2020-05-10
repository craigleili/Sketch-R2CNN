import os.path
import sys
import warnings

_project_folder_ = os.path.abspath('../')
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from scripts.base_train import SketchR2CNNTrain

if __name__ == '__main__':
    with SketchR2CNNTrain() as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()
