import os
import imgaug.augmenters as iaa
from glob import glob

# Root directory
ROOT_DIR = os.path.abspath('../')

# ======================================================================================================================

# Check before each training
EPOCHS = 200
TRAIN_LAYERS = '3+'  # 'heads', 'all'
DATASET_DIR = os.path.join(ROOT_DIR, 'data/dataset/20210817_CopyPaste_fullsize')
CSV_PATH = os.path.join(DATASET_DIR, f'log_{os.path.basename(DATASET_DIR)}.csv')

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'models', 'mask_rcnn_coco.h5')
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'models')
TRANS_WEIGHTS = os.path.join(ROOT_DIR, 'mask_rcnn_ichthyolith.h5')
AUGMENTATION = None # images are already augmented
# examples of augmentation can be checked by imaug_test.ipynb @ notebook
# AUGMENTATION = iaa.Sequential(
#     [
#      iaa.Fliplr(0.5),
#      iaa.Flipud(0.5), 
#      iaa.Rot90([0, 1]), # randome rotation
#      iaa.Affine(scale = {'x':(0.7, 2), 'y':(0.7, 2)},
#                 translate_px = {'x':(-100, 100), 'y':(-100, 100)},
#                 rotate = (0, 180),
#                 mode = 'constant', 
#                 cval = 0
#                ),
#      iaa.AdditiveGaussianNoise(scale=[15, 20]), # random noise
#     #  iaa.AddToBrightness((-20, 10)),
#     ])
SAVE_LOSS_FIGURE_PATH = os.path.join(DATASET_DIR, 'loss_curve.png')
OUTPUT_CSV = True
CLASS_WEIGHT = {1: 87, 2: 240, 3: 163} # not used in any process, change mrcnn/mode.py if you are to set class_weight
USE_CLASS_WEIGHT = False

# ======================================================================================================================

# Check before validation (select best model)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TEST_MODEL_DIR_NAME = 'latest' # 'latest' or dir_name (eg: 'ichthyolith20210226T2305')

if TEST_MODEL_DIR_NAME == 'latest':
    dirs = [dir_path for dir_path in glob(os.path.join(MODEL_DIR, '*')) if os.path.isdir(dir_path)]
    TEST_MODEL_DIR = max(dirs, key = os.path.getctime)
    TEST_MODEL_DIR_NAME = os.path.basename(TEST_MODEL_DIR)
else:
    TEST_MODEL_DIR = os.path.join(MODEL_DIR, TEST_MODEL_DIR_NAME)

NUM_CLASSES = 2  # change the value according to the number of classes (without 'bg')

DEVICE = '/gpu:0'  # /cpu:0 or /gpu:0
VALID_START_EPOCH = 1
# VALID_END_EPOCH = 40
VALID_END_EPOCH = EPOCHS + 0
VALID_CSV_DIR = os.path.join(ROOT_DIR, 'ichthyolith', 'valid_csv')
VALID_CSV_NAME = f'mAPs_{TEST_MODEL_DIR_NAME}.csv'
VALID_IOU_THRESHOLD = 0.5
VALID_MODE = 'box'  # 'box' or 'mask'
VALID_STEPS = 1
SAVE_INTERVAL = 5
IS_INTERRUPT = False

# ======================================================================================================================

# Check before testing (save detection results)
# TEST_DATASET_DIR = os.path.join(ROOT_DIR, 'data/dataset/original/test_dataset')
TEST_DATASET_DIR = os.path.join(ROOT_DIR, 'data/dataset/original/test_dataset')
TEST_MODEL_NAME = 'mask_rcnn_ichthyolith_0080.h5'
TEST_MODEL_PATH = os.path.join(TEST_MODEL_DIR, TEST_MODEL_NAME)
TEST_SUBDIR = 'test'
_OUT_DIR = os.path.join(ROOT_DIR, 'ichthyolith', 'out', TEST_MODEL_DIR_NAME)
DETECTIONS_PATH = os.path.join(_OUT_DIR, 'detections_20210215.pkl')
GTS_PATH = os.path.join(_OUT_DIR, 'gts_20210215.pkl')
OUTPUT_RESULTS = False
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
OUTPUT_RESULTS_ROOT_DIR = os.path.join(ROOT_DIR, 'ichthyolith', 'results', TEST_MODEL_DIR_NAME)
SAVE_GTS = True

# ======================================================================================================================
DET_DATASET_DIR = os.path.join(ROOT_DIR, 'samples/detection_test')
