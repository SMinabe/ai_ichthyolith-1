import os
import sys
import csv

import numpy as np
import tensorflow as tf

# ROOT directory of the project
ROOT_DIR = os.path.abspath('../')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from functions import compute_ap
from ichthyolith import ichthyolith_setting
from ichthyolith.ichthyolith_utils import get_detections_and_gts
from ichthyolith.ichthyolith_const import MODEL_DIR, DATASET_DIR, TEST_MODEL_DIR, NUM_CLASSES,\
    DEVICE, VALID_START_EPOCH, VALID_END_EPOCH, VALID_CSV_DIR, VALID_CSV_NAME, VALID_IOU_THRESHOLD, VALID_MODE,\
    VALID_STEPS, SAVE_INTERVAL, IS_INTERRUPT

if __name__ == '__main__':
    config = ichthyolith_setting.InferenceConfig()
    config.display()

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    TEST_MODE = 'inference'

    dataset = ichthyolith_setting.IchthyolithDataset()
    dataset.load_ichthyolith(DATASET_DIR, 'val')

    # Must call before using the dataset
    dataset.prepare()

    print(f'Image: {len(dataset.image_ids)}\nClasses: {dataset.class_names}')

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR,
                                  config=config)

    weights_paths = [i for i in os.listdir(TEST_MODEL_DIR) if '.h5' in i]

    weights_paths = sorted(weights_paths, key=lambda x: int(x[-7:-3]))
    weights_paths = weights_paths[VALID_START_EPOCH-1:VALID_END_EPOCH:VALID_STEPS]

    epoch_ids = list(range(VALID_START_EPOCH, VALID_END_EPOCH + 1, VALID_STEPS))
    save_csv_name = VALID_CSV_NAME
    save_cdv_dir = VALID_CSV_DIR
    os.makedirs(save_cdv_dir, exist_ok=True)
    save_csv_path = os.path.join(save_cdv_dir, save_csv_name)

    mAPs = []
    precisions = []
    recalls = []
    num_classes = NUM_CLASSES  # change
    cycle = 0

    if not IS_INTERRUPT:
        first_row = ['epoch', 'mAP']
        for i in range(1, num_classes+1):
            first_row.append(f'recall (class {i})')
            first_row.append(f'precision (class {i})')
        first_row.append('AR')
        with open(save_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(first_row)

    for num_weight, weights in enumerate(weights_paths):
        # Load weights
        weights_path = os.path.join(TEST_MODEL_DIR, weights)
        print('Loading weights ', weights_path)
        model.load_weights(weights_path, by_name=True)

        detections, gts = get_detections_and_gts(model, config, dataset)

        # Compute AP
        try:
            AP_list, precision_list, recall_list = [], [], []
            for label_id in range(1, num_classes+1):
                AP, p, r, _, _, _ = compute_ap.evaluate(det=detections, gt=gts, label=label_id,
                                               iou_threshold=VALID_IOU_THRESHOLD, mode=VALID_MODE)
                AP_list.append(AP)
                precision_list.append(p[-1])
                recall_list.append(r[-1])

            mAP = np.sum(AP_list) / len(AP_list)
            precisions.append(precision_list)
            recalls.append(recall_list)
            mAPs.append(mAP)
            for i in range(1, num_classes+1):
                print(f'class{i:2d}: {AP_list[i-1]}')
            print('mAP:', mAP)
            cycle += 1
            if cycle == SAVE_INTERVAL or num_weight == len(weights_paths)-1:
                with open(save_csv_path, 'a') as f:
                    writer = csv.writer(f)
                    for i, m, r, p in zip(epoch_ids, mAPs, recalls, precisions):
                        row = [i, m]
                        for c in range(num_classes):
                            row.append(r[c])
                            row.append(p[c])
                        row.append(np.mean(r))
                        writer.writerow(row)
                cycle = 0
                # print(f'saved EPOCH {epoch_ids[0]} ~ {epoch_ids[SAVE_INTERVAL-1]}')
                epoch_ids = epoch_ids[SAVE_INTERVAL:]
                mAPs = []
                recalls = []
                precisions = []
        except Exception as e:
            print(e)
            pass
