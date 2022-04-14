import os
import sys
import time
import math
import datetime
from glob import glob
import numpy as np

import cv2
import tensorflow as tf
from skimage.io import imread

from mrcnn import model as modellib
from mrcnn.config import Config
from functions.utils import save_pickle, masks2points_list


# ignore_classids を追加し，特定のクラスの検出結果を除外できるようにした (見邨)
# noise を検出すると数が大量になるので…
def detect(model, image, filename, ignore_classids=[], save_points = False, 
           rois_dtype = 'int16', save_mask = True):
    results = model.detect([image], verbose=0)
    result = results[0]
    result = {"filename": filename, "rois": result["rois"].astype(rois_dtype), "masks": result["masks"],
              "class_ids": result["class_ids"].astype("int8"), "scores": result["scores"].astype("float16")}
    # Exclude the detection result for the ignore_classids
    if len(ignore_classids) >= 1:
        not_ignored = np.isin(result['class_ids'], ignore_classids, invert=True)
        result['rois'] = result['rois'][not_ignored]
        result['masks'] = result['masks'][:, :, not_ignored]
        result['class_ids'] = result['class_ids'][not_ignored]
        result['scores'] = result['scores'][not_ignored]    
    
    # save contour by points
    if save_points:
        all_points_x_list, all_points_y_list = masks2points_list(result['masks'])
        result['all_points_xs'] = all_points_x_list
        result['all_points_ys'] = all_points_y_list
    if not save_mask:
        del result['masks']

    return result




# TODO: mask ではなく all_points で保存できるモードを作る

def detect_whole_slide(model, slide_image, crop_h, crop_w, stride, slide_name='', 
                       ignore_classids=[], save_points = False, save_mask = True):
    num_iter = 0
    detections = []
    slide_h, slide_w, _ = slide_image.shape
    # TODO: 右と下の端の漏れを修正
    # --> iter_h, iter_w に 1 を加え，スライド画像の右側と下側に白い領域を追加する？（見邨）

    iter_h = math.floor((slide_h - crop_h + stride) / stride)
    iter_w = math.floor((slide_w - crop_w + stride) / stride)
    for ih in range(iter_h):
        for jw in range(iter_w):
            start_h = ih * stride
            start_w = jw * stride
            crop_image = slide_image[start_h: start_h+crop_h, start_w: start_w+crop_w]
            # filename = slide_name.replace('.jpg', f'_{str(start_h).zfill(2)}_{str(start_w).zfill(2)}.jpg')
            filename = slide_name.replace('.jpg', f'_{start_h}_{start_w}.jpg') # zfill を削除
            result = detect(model, crop_image, filename, ignore_classids=ignore_classids, 
                            save_points = save_points, save_mask=save_mask)
            if len(result['rois']) > 0:
                detections.append([result])
            num_iter += 1
            sys.stdout.write(f"\r{num_iter}/{iter_h*iter_w}")
            sys.stdout.flush()
            time.sleep(0.01)
    return detections


def batch_detect_whole_slide(weights_path, slide_path_list, config, save_dir,
                             device='/gpu:0', overlap=0.9, time_output=False, ignore_classids = [], 
                             save_points = False, save_mask = True):
    with tf.device(device):
        model = modellib.MaskRCNN(mode='inference', model_dir='../models',
                                  config=config)

    print('Loading weights', weights_path)
    model.load_weights(weights_path, by_name=True)
    output_path = ''
    os.makedirs(save_dir, exist_ok=True)
    if time_output:
        now = datetime.datetime.now()
        output_path = os.path.join(save_dir, f'elapsed_time_{now:%Y%m%d%T%H%M}.txt')
        with open(output_path, 'w') as f:
            print(f'filename                           elapsed_time', file=f)
    whole_start_time = time.time()
    for slide_path in slide_path_list:
        slide_image = imread(slide_path)
        print(f'\nstart detecting.\nslide name: {os.path.basename(slide_path)}')

        start = time.time()
        detection = detect_whole_slide(model, slide_image, config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM,
                                       int(config.IMAGE_MIN_DIM * overlap), slide_name=os.path.basename(slide_path), 
                                       ignore_classids=ignore_classids, save_points = save_points, 
                                       save_mask=save_mask)
        elapsed_time = time.time() - start
        print()
        print(f'elapsed time: {int(round(elapsed_time))} [sec]') # 小数点以下切り捨て（見邨）

        # 検出の終了時間を予測するお節介機能を追加（見邨）
        # todo: 所要時間ではなく時刻を表示する --> datetime
        if slide_path == slide_path_list[0]:
            if len(slide_path_list) >= 2:
                time_required = elapsed_time * (len(slide_path_list) - 1) / 60
                print(f'estimated time required: {int(round(time_required))} [min]') 

        save_path = os.path.join(save_dir, os.path.basename(slide_path).replace('jpg', 'pkl'))
        save_pickle(detection, save_path)
        print(f'saved: {save_path}')
        if time_output:
            with open(output_path, 'a') as f:
                print(f'{os.path.basename(slide_path):<35}{elapsed_time}', file=f)
    whole_elapsed_time = time.time() - whole_start_time
    print(f'\nwhole elapsed time: {int(round(whole_elapsed_time))} [sec]')
    if time_output:
        with open(output_path, 'a') as f:
            print(f'---------------------------------------------------------', file=f)
            print(f'total                         {whole_elapsed_time}', file=f)



# 見邨追記
# 顕微鏡で撮影した分割画像を検出
# 画像名は，SampleName_slideNo_absX_absY.jpg
def batch_detect_microscopes(weights_path, microscope_img_dir, config, save_dir,
                             device='/gpu:0', time_output=False, ignore_classids = [], 
                             save_points = False, save_mask = True, slide_No = None):
    with tf.device(device):
        model = modellib.MaskRCNN(mode='inference', model_dir='../models',
                                  config=config)

    print('Loading weights', weights_path)
    model.load_weights(weights_path, by_name=True)
    output_path = ''
    os.makedirs(save_dir, exist_ok=True)
    if time_output:
        now = datetime.datetime.now()
        output_path = os.path.join(save_dir, f'elapsed_time_{now:%Y%m%d%T%H%M}.txt')
        with open(output_path, 'w') as f:
            print(f'filename                           elapsed_time', file=f)
    whole_start_time = time.time()
    
    # img_paths = glob(os.path.join(microscope_img_dir, '*.jpg'))
    img_paths = glob(f"{microscope_img_dir}/**/*.jpg", recursive=True)

    slide_names = list(set(['_'.join(os.path.basename(img_path).split('_')[:-2]) for img_path in img_paths]))
    if type(slide_No) == list:
        slide_names = [sn for sn in slide_names if int(sn.split('_')[-1]) in slide_No]
    slide_names.sort()
    print('following slide will be detected')
    print(slide_names)
    # img_temp = imread(img_paths[0])
    # stride = img_temp.shape[0]
    for slide_name in slide_names:
        print(f'\nstart detecting.\nslide name: {slide_name}')
        start = time.time()
        

        # detect_whole_slide を修正
        num_iter = 0
        detections = []
        # slide_h, slide_w, _ = slide_image.shape
        slide_img_paths = [path for path in img_paths if slide_name in os.path.basename(path)]
        for img_path in slide_img_paths:
            filename = os.path.basename(img_path)
            crop_image = imread(img_path)
            result = detect(model, crop_image, filename, ignore_classids=ignore_classids, 
                            save_points = save_points, rois_dtype = 'int32', 
                            save_mask=save_mask)
            if len(result['rois']) > 0:
                detections.append([result])
            if num_iter % 10 == 0:
                sys.stdout.write(f"\r{num_iter}/{len(slide_img_paths)}")
                sys.stdout.flush()
            num_iter += 1

        
        # iter_h = math.floor((slide_h - crop_h + stride) / stride)
        # iter_w = math.floor((slide_w - crop_w + stride) / stride)
        # for ih in range(iter_h):
        #     for jw in range(iter_w):
        #         start_h = ih * stride
        #         start_w = jw * stride
        #         crop_image = slide_image[start_h: start_h+crop_h, start_w: start_w+crop_w]
        #         # filename = slide_name.replace('.jpg', f'_{str(start_h).zfill(2)}_{str(start_w).zfill(2)}.jpg')
        #         filename = slide_name.replace('.jpg', f'_{start_h}_{start_w}.jpg') # zfill を削除
        #         result = detect(model, crop_image, filename, ignore_classids=ignore_classids, save_points = save_points)
        #         if len(result['rois']) > 0:
        #             detections.append([result])

        # detection = detect_whole_slide(model, slide_image, config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM,
        #                                int(config.IMAGE_MIN_DIM * overlap), slide_name=os.path.basename(slide_path), 
        #                                ignore_classids=ignore_classids, save_points = save_points)
        
        elapsed_time = time.time() - start
        print()
        print(f'elapsed time: {int(round(elapsed_time))} [sec]') # 小数点以下切り捨て（見邨）
        
        # # 終了時間の見積もり
        # if slide_path == slide_path_list[0]:
        #     if len(slide_path_list) >= 2:
        #         time_required = elapsed_time * (len(slide_path_list) - 1) / 60
        #         print(f'estimated time required: {int(round(time_required))} [min]') 

        save_path = os.path.join(save_dir, f'{slide_name}.pkl')
        save_pickle(detections, save_path)
        print(f'saved: {save_path}')
        if time_output:
            with open(output_path, 'a') as f:
                print(f'{slide_name:<35}{elapsed_time}', file=f)
    whole_elapsed_time = time.time() - whole_start_time
    print(f'\nwhole elapsed time: {int(round(whole_elapsed_time))} [sec]')
    if time_output:
        with open(output_path, 'a') as f:
            print(f'---------------------------------------------------------', file=f)
            print(f'total                         {whole_elapsed_time}', file=f)