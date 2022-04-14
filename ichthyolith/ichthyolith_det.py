import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath('../')

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import visualize
from mrcnn import model as modellib

from ichthyolith import ichthyolith_setting
from ichthyolith.ichthyolith_utils import get_ax
from ichthyolith.ichthyolith_const import MODEL_DIR, DET_DATASET_DIR, TEST_MODEL_PATH, DEVICE


if __name__ == '__main__':
    TEST_MODE = 'inference'
    config = ichthyolith_setting.InferenceConfig()
    config.display()

    dataset = ichthyolith_setting.IchthyolithDataset()
    dataset.load_ichthyolith(DET_DATASET_DIR, 'real')

    # Must call before using the dataset
    dataset.prepare()

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR,
                                  config=config)
    weights_path = TEST_MODEL_PATH
    # Load weights
    print('Loading weights ', weights_path)
    model.load_weights(weights_path, by_name=True)

    image_ids = dataset.image_ids
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        info = dataset.image_info[image_id]

        # Run object detection
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], ax=ax,
                                    title='Predictions')
