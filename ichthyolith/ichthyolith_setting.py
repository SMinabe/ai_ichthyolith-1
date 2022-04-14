import os
import sys
import json

import numpy as np
import skimage.io
import skimage.draw

"""
見邨追記
check followings before each training:
- NUM_CLASSES
- STEPS_PER_EPOCH
- VALIDATION_STEPS
- self.add_class()

"""


# Root directory of the project
ROOT_DIR = os.path.abspath('../')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class IchthyolithConfig(Config):
    """Configuration for training on the ichthyolith dataset.
    Derives from the base Config class and overrides som values
    """
    # Give the configuration a recognizable name
    NAME = 'ichthyolith'

    # We use GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3  # check train 'heads' or 'all'

    # Number of the classes (including background)
    NUM_CLASSES = 1 + 2  # Background + ichthyolith_classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = int(1600 / (IMAGES_PER_GPU * 1))  # the number of train images / batch_size (= IMAGES_PER_GPU * GPU_COUNT)
    
    # Number of validation steps per epoch
    VALIDATION_STEPS = 400 // IMAGES_PER_GPU + 1  # the number of valid images / batch_size (=IMAGES_PER_GPU * GPU_COUNT)

    # Skip detections with < DETECTION_MIN_CONFIDENCE
    DETECTION_MIN_CONFIDENCE = 0

    # IMAGE_MIN_DIM = 640
    # IMAGE_MAX_DIM = 640
    IMAGE_MIN_DIM = 1152
    IMAGE_MAX_DIM = 1152

    MAX_GT_INSTANCES = 75
 
    TRAIN_ROIS_PER_IMAGE = 600


class InferenceConfig(IchthyolithConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGE_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class IchthyolithDataset(utils.Dataset):

    def load_ichthyolith(self, dataset_dir, subset):
        """Load a subset of the ichthyolith dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to Load: train or val or test or real
        """

        # Add classes depending on the number of classes

        # add_class
        self.add_class('ichthyolith', 1, 'tooth')
        self.add_class('ichthyolith', 2, 'noise')
        
        # self.add_class('ichthyolith', 1, 'sharp_b_manual')
        # self.add_class('ichthyolith', 2, 'sharp_w_manual')
        # self.add_class('ichthyolith', 3, 'dull_b_manual')
        # self.add_class('ichthyolith', 4, 'dull_w_manual')
        # self.add_class('ichthyolith', 5, 'noise')     
        
        assert subset in ['train', 'val', 'test', 'real']
        dataset_dir = os.path.join(dataset_dir, subset)
        print(dataset_dir)
        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, 'via_region_data.json')))
        # annotations = json.load(open(os.path.join(dataset_dir, 'tooth_noise.json')))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations]

        # Add images
        for a in annotations:
            # Get the x, coordinates of points of the polygons the make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            animals = [r['region_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygonts to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the dataset is tiny
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                'ichthyolith',
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                animals=animals
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
            class_ids: a 1D array of class IDS of the instance masks.

            image_info = {
                'id': image_id,
                'source': source,
                'path': path,
                }
        """
        # If not a ichthyolith dataset image. delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info['source'] != 'ichthyolith':
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])],
                        dtype=np.uint8)  # Generate the number of mask
        for i, p in enumerate(info['polygons']):
            # Get indexes of pixels inside the polygons and set the to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr = np.where(rr >= 640, 639, rr)
            cc = np.where(cc >= 640, 639, cc)
            mask[rr, cc, i] = 1

        names = []
        for a in info['animals']:
            names.append(a['name'])

        class_ids = np.array([self.class_names.index(n) for n in names])
        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == 'ichthyolith':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)
