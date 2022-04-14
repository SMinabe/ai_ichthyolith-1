import os
import imgaug.augmenters as iaa

# ROOT directory
ROOT_DIR = os.path.abspath('../')

# ==================================================================================================================== #

# Check before training
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'models', 'mask_rcnn_coco.h5')
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'models')
DATASET_DIR = os.path.join(ROOT_DIR, 'data', 'dataset', 'dataset_tooth_denticle_without_handpicked')
TRANS_WEIGHTS = os.path.join(ROOT_DIR, 'mask_rcnn_ichthyolith.h5')

EPOCHS = 80
TRAIN_LAYERS = '3+'  # 'heads', '4+', '3+', 'all'
AUGMENTATION = iaa.Sequential(
    [iaa.Fliplr(0.5),
     iaa.Flipud(0.5)]
)

SAVE_LOSS_FIGURE_PATH = os.path.join(DATASET_DIR, 'loss_curve.png')
OUTPUT_CSV = True
CSV_PATH = os.path.join(DATASET_DIR, 'log_20210128.csv')
USE_CLASS_WEIGHTS = False
CLASS_WEIGHTS = {1: 3500, 2: 500}

# ==================================================================================================================== #

# Check before validation (select best model)
# Directory to save logs adn trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TEST_MODEL_DIR_NAME = 'logs_dataset_tooth_denticle_without_handpicked_20201030'
TEST_MODEL_DIR = os.path.join(MODEL_DIR, TEST_MODEL_DIR_NAME)
NUM_CLASSES = 2  # change the value according to the number of classes
DEVICE = '/gpu:0'  # /cpu:0 or /gpu:0
VALIDATION_START_EPOCH = 1
VALIDATION_END_EPOCH = 80
VALID_CSV_DIR = os.path.join(ROOT_DIR, 'results', 'valid_csv')
VALID_CSV_NAME = f'mAPs_{TEST_MODEL_DIR_NAME}.csv'
VALID_IOU_THRESHOLD = 0.5
VALID_MODEL = 'box'
VALID_STEPS = 1
SAVE_INTERVAL = 5
IS_INTERRUPT = False


# ==================================================================================================================== #

# check before testing (save detection results)
TEST_DATASET_DIR = os.path.join(ROOT_DIR, 'data', 'dataset', 'original', 'test_dataset')
TEST_MODEL_NAME = 'mask_rcnn_ichthyolith_0080.h5'
TEST_MODEL_PATH = os.path.join(TEST_MODEL_DIR, TEST_MODEL_NAME)
TEST_SUBDIR = 'test'
_OUT_DIR = os.path.join(ROOT_DIR, 'results', TEST_MODEL_DIR_NAME, 'test')
DETECTIONS_PATH = os.path.join(_OUT_DIR, 'detections_0080.pkl')
GTP_PATH = os.path.join(_OUT_DIR, 'gts.pkl')
OUTPUT_RESULTS = False
THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
SAVE_GTS = False


# ==================================================================================================================== #


IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
