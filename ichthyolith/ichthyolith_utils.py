import os
import sys
import pickle

import cv2
import numpy as np
import skimage.io
import skimage.color
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath('../')
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import model as modellib


def color_splash(image, mask):
    """Apply color splach effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask info one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original as color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print(f'Running on {image_path}')
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = 'splash_' + os.path.basename(image_path).replace('jpg', 'png')
        skimage.io.imsave(file_name, splash)
        print('saved to ', file_name)
    elif video_path:
        # Video capture
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = 'splash_' + os.path.basename(video_path)  # .avi
        video_writer = cv2.VideoWriter(file_name,
                                       cv2.VideoWriter_fourcc(*'MJPG'),
                                       fps, (width, height))
        count = 0
        success = True
        while success:
            print('frame: ', count)
            # Read next image
            success, image = cap.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                video_writer.write(splash)
                count += 1
        video_writer.release()
        print('saved to ', file_name)


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualization in the notebook. Provided a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def get_detections_and_gts(model, config, dataset):
    num_iter = 0
    detections = []
    gts = []
    image_ids = dataset.image_ids
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        r0 = results[0]
        filename = dataset.image_info[image_id]['id']
        r0 = {"filename": filename, "rois": r0["rois"].astype("int16"), "masks": r0["masks"],
              "class_ids": r0["class_ids"].astype("int8"), "scores": r0["scores"].astype("float16")}

        detections.append([r0])
        gts.append([{"filename": filename, "gt_bbox": gt_bbox.astype("int16"),
                     "gt_class_id": gt_class_id.astype("int8"), "gt_mask": gt_mask.astype("int8")}])
        num_iter += 1
        print(f'{num_iter}/{len(image_ids)}')
    return detections, gts


def save_detections_and_gts(detections, gts, save_detections_path, save_gts_path):
    os.makedirs(os.path.dirname(save_detections_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_gts_path), exist_ok=True)
    with open(save_detections_path, 'wb') as f:
        pickle.dump(detections, f, protocol=-1)
    with open(save_gts_path, 'wb') as f:
        pickle.dump(gts, f, protocol=-1)


def save_pickle(obj, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, protocol=-1)
