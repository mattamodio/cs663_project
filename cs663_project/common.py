import os

import numpy as np


#### FILEPATHS ####
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '..'
))
DATA_ROOT = PROJECT_ROOT + 'data/'
DATA_RAW = DATA_ROOT + 'raw/'
DATA_PROCESSED = DATA_ROOT + 'proc/'
MODEL_FOLDER = PROJECT_ROOT + '_models/'
FIGURE_FOLDER = PROJECT_ROOT + '_figures/'