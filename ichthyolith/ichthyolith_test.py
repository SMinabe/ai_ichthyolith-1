import os
import sys
import csv
import pickle
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath('../')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib

from functions import compute_ap
from ichthyolith import ichthyolith_setting
from ichthyolith.ichthyolith_utils import get_detections_and_gts, save_detections_and_gts, save_pickle
from ichthyolith.ichthyolith_const import MODEL_DIR, TEST_DATASET_DIR, DEVICE, NUM_CLASSES, \
    TEST_MODEL_PATH, DETECTIONS_PATH, GTS_PATH, OUTPUT_RESULTS, THRESHOLDS, OUTPUT_RESULTS_ROOT_DIR,\
    VALID_MODE, TEST_SUBDIR, SAVE_GTS


if __name__ == '__main__':
    config = ichthyolith_setting.InferenceConfig()
    config.display()

    TEST_MODE = 'inference'

    dataset = ichthyolith_setting.IchthyolithDataset()
    dataset.load_ichthyolith(TEST_DATASET_DIR, TEST_SUBDIR)

    # Must call before using the dataset
    dataset.prepare()

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR,
                                  config=config)

    # set test weights
    weights_path = TEST_MODEL_PATH

    # Load weights
    print('Loading weights', weights_path)
    model.load_weights(weights_path, by_name=True)

    detections, gts = get_detections_and_gts(model, config, dataset)

    # save detections and gts
    save_detections_path = DETECTIONS_PATH
    save_gts_path = GTS_PATH
    save_pickle(detections, save_detections_path)
    if SAVE_GTS:
        save_pickle(gts, save_gts_path)
    # save_detections_and_gts(detections, gts, save_detections_path, save_gts_path)
    print('saved detections and gts')

    if OUTPUT_RESULTS:
        # Compute AP
        num_classes = NUM_CLASSES
        thresholds = THRESHOLDS
        save_root_dir = OUTPUT_RESULTS_ROOT_DIR
        os.makedirs(save_root_dir, exist_ok=True)
        for threshold in thresholds:
            AP_list = []
            for label_id in range(1, num_classes+1):
                AP, precisions, recalls, _, _, _ = compute_ap.evaluate(det=detections, gt=gts, label=label_id,
                                                                       iou_threshold=threshold, mode=VALID_MODE)
                AP_list.append(AP)
                save_csv_path = os.path.join(save_root_dir, f'group{label_id:02d}/{threshold}_group{label_id:02d}.csv')
                os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
                with open(save_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([os.path.basename(weights_path), datetime.datetime.now()])
                    writer.writerow(['Recall', 'Precision'])
                    for precision, recall in zip(precisions, recalls):
                        writer.writerow([recall, precision])
                    writer.writerow(['AP', AP])
            mAP = np.sum(AP_list) / len(AP_list)
            print(f'mAP: {mAP}, (threshold={threshold})')
