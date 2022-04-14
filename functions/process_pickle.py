import os
import pickle
from glob import glob

import numpy as np

from mrcnn.utils import non_max_suppression
from functions.utils import load_pickle, save_pickle


def get_top_left_coordinate(filename):
    parts = filename.split('_')
    return int(parts[-2]), int(parts[-1].replace('.jpg', ''))


def get_original_slide_name(filename):
    parts = filename.split('_')
    slide_name = '_'.join(parts[:-2]) + '.jpg'
    return slide_name


def convert_to_abs_rois(rois, x, y):
    """
    :param rois: [N, (y1, x1, y2, x2)]
    :param x: x coordinate of image top left
    :param y: y coordinate of image top left
    :return abs_rois: [N, (y1', x2', y2', x2')] converted to absolute coordinates
    """
    rois[:, 0] = rois[:, 0] + y
    rois[:, 1] = rois[:, 1] + x
    rois[:, 2] = rois[:, 2] + y
    rois[:, 3] = rois[:, 3] + x
    return rois

def convert_to_rel_rois(rois, x, y):
    rois[:, 0] = rois[:, 0] - y
    rois[:, 1] = rois[:, 1] - x
    rois[:, 2] = rois[:, 2] - y
    rois[:, 3] = rois[:, 3] - x
    return rois

def convert_to_abs_points(all_points_xs, all_points_ys, x, y): # 見
    """
    :param all_points_xs: [np.array([x1, x2, x3, ...]), np.array([x1, x2, x3, ...]), np.array([x1, x2, x3, ...]), ...]
    :param all_points_ys: [np.array([y1, y2, y3, ...]), np.array([y1, y2, y3, ...]), np.array([y1, y2, y3, ...]), ...]
    :param x: x coordinate of image top left
    :param y: y coordinate of image top left
    :return abs_all_points_xs, abs_all_points_ys
    """
    abs_all_points_xs, abs_all_points_ys = [], []
    for i, all_points_x in enumerate(all_points_xs):
        all_points_y = all_points_ys[i]
        abs_all_points_xs.append(np.array(all_points_x) + x)
        abs_all_points_ys.append(np.array(all_points_y) + y)
    return(abs_all_points_xs, abs_all_points_ys)

def convert_to_rel_points(abs_all_points_xs, abs_all_points_ys, x, y): # 見
    rel_all_points_xs, rel_all_points_ys = [], []
    for i, abs_all_points_x in enumerate(abs_all_points_xs):
        abs_all_points_y = abs_all_points_ys[i]
        rel_all_points_xs.append(np.array(abs_all_points_x) - x)
        rel_all_points_ys.append(np.array(abs_all_points_y) - y)
    return(abs_all_points_xs, abs_all_points_ys)


# TODO: fix
def convert_to_eval_format(pickle_path, IoU=0.5, rois_dtype = 'int16', save_croppedName = False):
    convertAbsPoints = False 
    pickle_data = load_pickle(pickle_path)
    slide_name = ''
    new_rois = np.empty((0, 4), dtype=rois_dtype)
    new_class_ids = np.array([], dtype=np.int8)
    new_scores = np.array([], dtype=np.float16)
    new_all_points_xs, new_all_points_ys = [], []
    cropped_names = []

    for r in pickle_data:
        slide_name = get_original_slide_name(r[0]['filename'])
        y, x = get_top_left_coordinate(r[0]['filename'])
        
        # print(f'y: {y}')
        # print(f'x: {x}')
        # print(f'before: {r[0]["rois"]}')
        abs_rois = convert_to_abs_rois(r[0]['rois'], x, y)
        # print(f'after : {r[0]["rois"]}')
        
        new_rois = np.vstack((new_rois, abs_rois))
        new_class_ids = np.append(new_class_ids, r[0]['class_ids'])
        new_scores = np.append(new_scores, r[0]['scores'])
        
        # points の変換 (見)
        if 'all_points_xs' in r[0].keys():
            convertAbsPoints = True
            abs_all_points_xs, abs_all_points_ys = convert_to_abs_points(r[0]['all_points_xs'], r[0]['all_points_ys'], x, y)
            new_all_points_xs += abs_all_points_xs
            new_all_points_ys += abs_all_points_ys

        # 切り出し画像名の取得（見）
        if save_croppedName:
            num_classIds = len(r[0]['class_ids'])
            cropped_names += [r[0]['filename']] * num_classIds

    new_masks = np.zeros((1, 1, len(new_class_ids)), dtype=np.bool)
    pick = non_max_suppression(new_rois, new_scores, IoU)
    print()
    print(f'removed {new_rois.shape[0] - len(pick)} rois by non maximum suppression.')
    print(f'before: {new_rois.shape[0]}')
    new_rois = new_rois[pick, :]
    new_scores = new_scores[pick]
    new_class_ids = new_class_ids[pick]
    print(f'after : {new_rois.shape[0]}')
    
    eval_dict = {'filename': slide_name, 'rois': new_rois, 'masks': new_masks,
                 'class_ids': new_class_ids, 'scores': new_scores}
    if convertAbsPoints:
        new_all_points_xs = [new_all_points_xs[p] for p in pick]
        new_all_points_ys = [new_all_points_ys[p] for p in pick]
        eval_dict['all_points_xs'] = new_all_points_xs
        eval_dict['all_points_ys'] = new_all_points_ys

    if save_croppedName:
        new_cropped_names = [cropped_names[p] for p in pick]
        eval_dict['cropped_imgName'] = new_cropped_names
        # eval_format = [[{'filename': slide_name, 'rois': new_rois, 'masks': new_masks,
        #                  'all_points_xs':new_all_points_xs, 'all_points_ys':new_all_points_ys,
        #                  'class_ids': new_class_ids, 'scores': new_scores, 'cropped_imgName': cropped_imgName}]] 
    # else:
    #     eval_format = [[{'filename': slide_name, 'rois': new_rois, 'masks': new_masks,
    #                     'class_ids': new_class_ids, 'scores': new_scores, 'cropped_imgName': cropped_imgName}]]
    
    eval_format = [[eval_dict]]
    return eval_format


def batch_convert_to_eval_format(pickle_path_list, save_path=None, IoU=0.5, 
                                 save_croppedName = False, rois_dtype = 'int16'):
    detections = []
    for pickle_path in pickle_path_list:
        eval_format = convert_to_eval_format(pickle_path, save_croppedName = save_croppedName, 
                                             rois_dtype=rois_dtype, IoU=IoU)
        detections.append(eval_format[0])
    if save_path is not None:
        save_pickle(detections, save_path)
    print('created detections results for evaluation.')
    return detections


# TODO: cleanup
def extract_by_scores(detections, score_threshold):
    new_detections = []
    for i in range(len(detections)):
        filename = detections[i][0]['filename']
        scores = detections[i][0]['scores']
        masks = detections[i][0]['masks']
        index_over_score = np.where(scores >= score_threshold)[0]
        rois = detections[i][0]['rois'][index_over_score]
        class_ids = detections[i][0]['class_ids'][index_over_score]
        scores = scores[index_over_score]
        masks = masks[:, :, index_over_score]
        contents = {'filename': filename, 'rois': rois, 'masks': masks,
                    'class_ids': class_ids, 'scores': scores}
        for key in ['all_points_xs', 'all_points_ys', 'cropped_imgName']:
            if key in detections[i][0].keys():
                contents[key] = [detections[i][0][key][ios] for ios in index_over_score]
        new_detections.append([contents])
    return new_detections


def extract_specific_size(gts, min_size=None, max_size=None, size_regulation=640):
    new_gts = []
    for i in range(len(gts)):
        filename = gts[i][0]['filename']
        gt_bbox = gts[i][0]['gt_bbox']
        gt_class_id = gts[i][0]['gt_class_id']
        gt_mask = gts[i][0]['gt_mask']
        size = []
        for bbox in gt_bbox:
            s = ((bbox[2] - bbox[0]) / size_regulation) * ((bbox[3] - bbox[1]) / size_regulation)
            size.append(s)
        size = np.array(size)
        if (min_size is not None) and (max_size is not None):
            index = np.where((size > min_size) & (size < max_size))[0]
        elif (min_size is not None) and (max_size is None):
            index = np.where((size > min_size))[0]
        elif (min_size is None) and (max_size is not None):
            index = np.where((size < max_size))[0]
        else:
            index = np.where(size > 0)[0]
        gt_bbox = gt_bbox[index]
        gt_class_id = gt_class_id[index]
        if gt_mask.shape == 3:
            gt_mask = gt_mask[:, :, index]
        else:
            gt_mask = gt_mask[index]
        contents = {'filename': filename, 'gt_bbox': gt_bbox,
                    'gt_mask': gt_mask, 'gt_class_id': gt_class_id}
        new_gts.append([contents])
    return new_gts


def convert_detections_labels(detections, assign_dict):
    new_detections = []
    for i in range(len(detections)):
        class_ids = detections[i][0]['class_ids']
        class_ids = np.array([assign_dict[class_id] for class_id in class_ids])
        contents = {'filename': detections[i][0]['filename'], 'rois': detections[i][0]['rois'],
                    'masks': detections[i][0]['masks'], 'class_ids': class_ids,
                    'scores': detections[i][0]['scores']}

        for key in ['all_points_xs', 'all_points_ys', 'cropped_imgName']:
            if key in detections[i][0].keys():
                contents[key] = detections[i][0][key]

        new_detections.append([contents])
    return new_detections


def convert_gts_labels(gts, assign_dict):
    new_gts = []
    for i in range(len(gts)):
        gt_class_id = gts[i][0]['gt_class_id']
        gt_class_id = np.array([assign_dict[class_id] for class_id in gt_class_id])
        contents = {'filename': gts[i][0]['filename'], 'gt_bbox': gts[i][0]['gt_bbox'],
                    'gt_class_id': gt_class_id, 'gt_mask': gts[i][0]['gt_mask']}
        if 'gt_size' in gts[i][0].keys():
            contents['gt_size'] = gts[i][0]['gt_size']
            contents['gt_length'] = gts[i][0]['gt_length']
        
        new_gts.append([contents])
    return new_gts
