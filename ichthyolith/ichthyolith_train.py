import os
import sys
import warnings
import argparse

import keras
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
from mrcnn import model as modellib, utils
from ichthyolith import ichthyolith_setting
from ichthyolith.ichthyolith_const import COCO_WEIGHTS_PATH, DEFAULT_LOGS_DIR, \
    TRANS_WEIGHTS, EPOCHS, TRAIN_LAYERS, DATASET_DIR, AUGMENTATION, SAVE_LOSS_FIGURE_PATH, \
    OUTPUT_CSV, CSV_PATH, CLASS_WEIGHT, USE_CLASS_WEIGHT

warnings.simplefilter('ignore')

def train(model, dataset_dir, config, custom_callbacks=None):
    """Train the model."""
    # Training dataset
    dataset_train = ichthyolith_setting.IchthyolithDataset()
    dataset_train.load_ichthyolith(dataset_dir, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ichthyolith_setting.IchthyolithDataset()
    dataset_val.load_ichthyolith(dataset_dir, 'val')
    dataset_val.prepare()

    # *** This training schedule is an example. Update yo your needs ***
    print('Training net work heads')
    hist = model.train(dataset_train, dataset_val,
                       learning_rate=config.LEARNING_RATE,
                       epochs=EPOCHS,
                       layers=TRAIN_LAYERS,
                       augmentation=AUGMENTATION,
                       custom_callbacks=custom_callbacks)  # if only learning head, set 'heads'
    return hist


def plot_loss(hist, verbose=0, save_path=None):
    fig = plt.figure()
    plt.title('Mask R-CNN Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.legend()
    if verbose:
        plt.show()
    if not (save_path is None):
        fig.savefig(save_path)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ichthyolith.'
    )
    parser.add_argument('--dataset', required=False,
                        default=DATASET_DIR,
                        metavar='/path/to/ichthyolith/dataset',
                        help='Directory of the Ichthyolith dataset')
    parser.add_argument('--weights', required=True,
                        metavar='path/to/weights.h5',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar='/path/to/logs/',
                        help='Logs and checkpoints directory (default=models)')
    parser.add_argument('--image', required=False,
                        metavar='Path or URL to image',
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar='Path or URL to video',
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    assert args.dataset, 'Argument --dataset is required for training'

    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Logs: ', args.logs)

    # Configurations
    config = ichthyolith_setting.IchthyolithConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode='training', config=config,
                              model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == 'coco':
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == 'imagenet':
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif args.weights.lower == 'trans':
        weights_path = TRANS_WEIGHTS
    else:
        weights_path = args.weights

    # Load weights
    print('Loading weights ', weights_path)
    if args.weights.lower() == 'coco':
        # Exclude the las layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            'mrcnn_class_logits', 'mrcnn_bbox_fc',
            'mrcnn_bbox', 'mrcnn_mask'
        ])
    else:
        model.load_weights(weights_path, by_name=True)

    if OUTPUT_CSV:
        callbacks = [keras.callbacks.CSVLogger(CSV_PATH, separator=',', append=False)]
    else:
        callbacks = None

    if USE_CLASS_WEIGHT:
        history = train(model, args.dataset, config, custom_callbacks=callbacks)
    else:
        history = train(model, args.dataset, config, custom_callbacks=callbacks)
    # plot_loss(history, verbose=0, save_path=SAVE_LOSS_FIGURE_PATH)
