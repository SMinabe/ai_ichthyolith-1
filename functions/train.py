import os
import sys
import argparse

import keras
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
# Import Mask RCNN
from mrcnn import model as modellib, utils
from functions.constant import *


def train(model, dataset_dir, config, custom_callbacks=None):
    """Train the model."""
    # Training dataset
