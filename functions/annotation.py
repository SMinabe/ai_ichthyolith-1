import os
import shutil
from glob import glob
import cv2
import numpy as np
import pandas as pd
import random

from functions.process_excel import ExcelDataProcessor
from functions.process_json import convert_class_label
from functions.utils import load_json, load_pickle, save_json, load_xml, generate_json_template, save_pickle

# このファイル一式見邨作成


def remove_slide(ROOT_DIR, sample_ID, remove_slide_numbers, reasons):
    if len(remove_slide_numbers) >= 1:
        img_dir = os.path.dirname(glob(f'{ROOT_DIR}data/images/*/*/{sample_ID}*.jpg')[0])
        remove_dir = os.path.join(img_dir, 'not_for_learning')
        os.makedirs(remove_dir, exist_ok = True)
        
        for slide_img_path in glob(f'{ROOT_DIR}data/images/*/*/{sample_ID}*.jpg'):
            if int(os.path.basename(slide_img_path)[-7:-4]) in remove_slide_numbers:
                shutil.move(slide_img_path, remove_dir)
        
        # todo: describe reasons for removing each image file (in excel file?)       
        with open(os.path.join(remove_dir, 'reason.txt'), mode='w') as f:
            f.write(reasons)

def split_slide_from_directory(ROOT_DIR, sample_ID, stride = 640, saveNameMode = 'ij', 
                               select_slide = None, slide_img_dir = None):
    split_img_dir = f'{ROOT_DIR}data/annotations/{sample_ID}/01_split'
    refresh_directory(split_img_dir)
    if slide_img_dir is None:
        slide_img_dir = os.path.dirname(glob(f'{ROOT_DIR}data/images/*/*/{sample_ID}*.jpg')[0])
    for slide_img_path in glob(os.path.join(slide_img_dir, f'{sample_ID}*.jpg')):
        slide_img = cv2.imread(slide_img_path)
        slide_img_name = os.path.basename(slide_img_path)
        flag = True
        if type(select_slide) == list:
            if slide_img_name not in select_slide:
                flag = False
        if flag:
            print(slide_img_name)
            split_slide(slide_img, slide_img_name, split_img_dir, stride, saveNameMode = saveNameMode)

def split_slide_with_detection(ROOT_DIR, sample_ID, detection_excel_path, 
                               stride = 640, contour_color = (0, 255, 0), condition_dict = {}, 
                               save_excel_path = None, saveNameMode = 'ij'):
    slide_img_dir = os.path.dirname(glob(f'{ROOT_DIR}data/images/*/*/{sample_ID}*.jpg')[0])
    split_img_dir = f'{ROOT_DIR}data/annotations/{sample_ID}/11_split'
    refresh_directory(split_img_dir)
    df_detection = pd.read_excel(detection_excel_path, index_col = 0)
    for k, v in condition_dict.items():
        assert type(v) in [str, int, float, list]
        if type(v) == list:
            df_detection = df_detection[df_detection[k].isin(v)]
        else:
            df_detection = df_detection[df_detection[k] == v]

    if save_excel_path is not None:
        df_detection.to_excel(save_excel_path)
    count = 0
    slide_img_names = sorted(set(df_detection['original_slide_name']))
    for slide_img_name in slide_img_names:
        print(slide_img_name)
        slide_img_path = os.path.join(slide_img_dir, slide_img_name)
        slide_img = cv2.imread(slide_img_path)
        df_detection_slide = df_detection[df_detection['original_slide_name'] == slide_img_name]
        contours = []
        for index, item in df_detection_slide.iterrows():
            x_min, y_min = int(item['x_min']), int(item['y_min'])
            x_max, y_max = int(item['x_max']), int(item['y_max'])

            for x in [x_min, x_max]:
                slide_img[y_min:y_max, x] = contour_color
            for y in [y_min, y_max]:
                slide_img[y, x_min:x_max] = contour_color

        split_slide(slide_img, slide_img_name, split_img_dir, stride, saveNameMode = saveNameMode)

def split_slide_with_annotation(ROOT_DIR, sample_ID, detection_excel_path, 
                               stride = 640, contour_color = (0, 255, 0), condition_dict = {}, 
                               save_excel_path = None, saveNameMode = 'ij'):
    slide_img_dir = os.path.dirname(glob(f'{ROOT_DIR}data/images/*/*/{sample_ID}*.jpg')[0])
    split_img_dir = f'{ROOT_DIR}data/annotations/{sample_ID}/11_split'
    refresh_directory(split_img_dir)
    df_detection = pd.read_excel(detection_excel_path, index_col = 0)
    for k, v in condition_dict.items():
        assert type(v) in [str, int, float, list]
        if type(v) == list:
            df_detection = df_detection[df_detection[k].isin(v)]
        else:
            df_detection = df_detection[df_detection[k] == v]

    if save_excel_path is not None:
        df_detection.to_excel(save_excel_path)
    count = 0
    slide_img_names = sorted(set(df_detection['original_slide_name']))
    for slide_img_name in slide_img_names:
        print(slide_img_name)
        slide_img_path = os.path.join(slide_img_dir, slide_img_name)
        slide_img = cv2.imread(slide_img_path)
        df_detection_slide = df_detection[df_detection['original_slide_name'] == slide_img_name]
        contours = []
        
        for index, item in df_detection_slide.iterrows():
            x_min, y_min = int(item['x_min']), int(item['y_min'])
            x_max, y_max = int(item['x_max']), int(item['y_max'])

            for x in [x_min, x_max]:
                slide_img[y_min:y_max, x] = contour_color
            for y in [y_min, y_max]:
                slide_img[y, x_min:x_max] = contour_color

        split_slide(slide_img, slide_img_name, split_img_dir, stride, saveNameMode = saveNameMode)



def split_slide(slide_img, slide_img_name, save_dir, stride = 640, save_black = False, saveNameMode = 'ij'):  
    hh = slide_img.shape[0]
    ww = slide_img.shape[1]
    imax = hh // stride + 1
    jmax = ww // stride + 1

    for i in range(imax):
        for j in range(jmax):
            img1 = np.ones((stride, stride, 3), np.uint8) * 255
            img1[0:min(hh - stride * i, stride), 0:min(ww - stride * j, stride)] =\
                    slide_img[stride * i : min(hh, stride * (i + 1)), \
                        stride * j : min(ww, stride * (j + 1))]
            
            if any([save_black, len(set(img1.flatten())) >= 3]):
                if saveNameMode == 'yx':
                    save_name = '{}_{}_{}.jpg'.format(slide_img_name[:-4], i * stride, j * stride)
                else:
                    save_name = '{}_{:0=2}_{:0=2}.jpg'.format(slide_img_name[:-4], i, j)
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, img1)

def trim_objects(ROOT_DIR, trimmed_dir, xml_dir, sample_ID, class_list, excel_path, margin = None, slide_img_dir=None):
    df = pd.DataFrame()
    df.to_excel(excel_path)
    
    os.makedirs(trimmed_dir, exist_ok=True)

    for cl in class_list:
        trimmed_class_dir = os.path.join(trimmed_dir, cl)
        refresh_directory(trimmed_class_dir)
        edp = ExcelDataProcessor(excel_path)
        edp.load_from_xml(ROOT_DIR, xml_dir, trimmed_class_dir = trimmed_class_dir, margin = margin, slide_img_dir=slide_img_dir, cl_name = cl)


def xml2excel(xml_dir, excel_path, ignore_classes = [], stride = 640, mode = 'ij'):
    xml_path_list = glob(os.path.join(xml_dir, '*.xml'))
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, index_col=0)
    else:
        df = pd.DataFrame()
    try:
        index = max(df.index) + 1 
    except:
        index = 1
    
    for xml_path in xml_path_list:
        xml_name = os.path.basename(xml_path)
        if mode == 'xy':
            x_start = int(xml_name[:-4].split('_')[-1])
            y_start = int(xml_name[:-4].split('_')[-2])
        else:
            x_start = stride * int(xml_name[:-4].split('_')[-1])
            y_start = stride * int(xml_name[:-4].split('_')[-2])
        original_slide_name = '_'.join(xml_name[:-4].split('_')[:-2]) + '.jpg'
        class_name_list, x_min_list, y_min_list, x_max_list, y_max_list\
            = load_xml(xml_path)
        for i, cl in enumerate(class_name_list):
            if cl not in ignore_classes:
                df.loc[index, 'id'] = index
                df.loc[index, 'original_slide_name'] = original_slide_name
                df.loc[index, 'x_min'] = x_start + x_min_list[i]
                df.loc[index, 'y_min'] = y_start + y_min_list[i]
                df.loc[index, 'x_max'] = x_start + x_max_list[i]
                df.loc[index, 'y_max'] = y_start + y_max_list[i]
                df.loc[index, 'class'] = cl
                df.loc[index, 'cropped_img_name'] = xml_name[:-4] + '.jpg'
                index += 1
    df.to_excel(excel_path)


def xml2pickle(xml_dir, class_list, save_pickle_path):
    xml_path_list = glob(os.path.join(xml_dir, '*.xml'))
    data = []
    
    for xml_path in xml_path_list:
        # todo: convert to Minabe format
        xml_name = os.path.basename(xml_path)
        filename = xml_name[:-4] + '.jpg'
        class_name_list, x_min_list, y_min_list, x_max_list, y_max_list\
            = load_xml(xml_path)
        
        for i in range(len(class_name_list)):
            # rois: y1, x1, y2, x2
            if class_name_list[i] in class_list:
                rois = np.array([[y_min_list[i], x_min_list[i], y_max_list[i], x_max_list[i]]])
                class_id = class_list.index(class_name_list[i])
                score = 1.0
                result = {"filename": filename, "rois": rois.astype("int32"), 
                        "class_ids": np.array([class_id]).astype("int8"), 
                        "scores": np.array([score]).astype("float16")}
                data.append([result])
    save_pickle(data, save_pickle_path)

def pickle2gts(pickle_path):
    data = load_pickle(pickle_path)
    new_data = []
    for i in range(len(data)):
        pass


def json2excel(trimmed_dir, excel_path, class_list, pixel2micron = 2.54):
    
    # json の空きクラスを埋める処理は，fix_json に移行．問題がないか注意
    
    df = pd.read_excel(excel_path, index_col=0)
    
    for cl in class_list:
        trimmed_class_dir = os.path.join(trimmed_dir, cl)
        json_path = os.path.join(trimmed_class_dir, 'via_region_data.json')
        json_data = load_json(json_path)
        for k, v in json_data.items():
            identical_number = int((v['filename']).split('_')[0])
            df_id = df[df['id'] == identical_number]
            x_start = list(df_id['x_min'])[0]
            y_start = list(df_id['y_min'])[0]
            index = list(df_id.index)[0]

            if len(v['regions'].keys()) >= 1:
                all_points_x =  v['regions']['0']['shape_attributes']['all_points_x']
                all_points_y =  v['regions']['0']['shape_attributes']['all_points_y']
                size = _get_size(all_points_x, all_points_y)
                length = _get_length(all_points_x, all_points_y, pixel2micron = pixel2micron)
                all_points_x = np.array(all_points_x) + x_start
                all_points_y = np.array(all_points_y) + y_start
                x_min, y_min = min(all_points_x), min(all_points_y)
                x_max, y_max = max(all_points_x), max(all_points_y)
                all_points_x = _to_string(all_points_x)
                all_points_y = _to_string(all_points_y)
                class_name = cl
            else:
                x_min, y_min, x_max, y_max, all_points_x, all_points_y, size, length, class_name \
                    = [np.nan] * 9

            # write data to excel
            df.loc[index, 'x_min'] = x_min
            df.loc[index, 'y_min'] = y_min
            df.loc[index, 'x_max'] = x_max
            df.loc[index, 'y_max'] = y_max
            df.loc[index, 'all_points_x'] = all_points_x
            df.loc[index, 'all_points_y'] = all_points_y
            df.loc[index, 'size'] = size
            df.loc[index, 'length'] = length

            
        #     if len(v['regions'].keys()) >= 1:
        #         if 'name' not in v['regions']['0']['region_attributes'].keys():
        #             json_data[k]['regions']['0']['region_attributes'] = {'name': cl}

        # save_json(json_data, json_path)

    df.to_excel(excel_path)

def fix_json(json_path, class_name):
    # クラスラベルのない json data に特定のクラスを割り当てる
    json_data = load_json(json_path)
    for k1, v1 in json_data.items():
        for k2, v2 in v1['regions'].items():
            if 'name' not in v2['region_attributes'].keys():
                json_data[k1]['regions'][k2]['region_attributes'] = {'name': class_name}
    save_json(json_data, json_path)

def trim_img_for_noise_annotation(ROOT_DIR, annot_base_dir, sample_ID, excel_path, focus_class_list, stride = 640, margin = 320):
    df = pd.read_excel(excel_path)
    df = df[df['class'].isin(focus_class_list)]
    
    save_dir = os.path.join(annot_base_dir, '03_noise_labeling')
    refresh_directory(save_dir)
    
    slide_img_paths = glob(f'{ROOT_DIR}data/images/*/*/{sample_ID}*.jpg')
    for slide_img_path in slide_img_paths:
        slide_img = cv2.imread(slide_img_path)
        slide_img_name = os.path.basename(slide_img_path)
        df_img = df[df['original_slide_name'] == slide_img_name]

        mask = np.zeros_like(slide_img)
        for _, item in df_img.iterrows():
            x_min = max(item['x_min'] - margin, 0)
            y_min = max(item['y_min'] - margin, 0)
            x_max = min(item['x_max'] + margin, mask.shape[1])
            y_max = min(item['y_max'] + margin, mask.shape[0])
            mask[y_min:y_max, x_min:x_max] = np.array([1, 1, 1])
        
        masked_slide_img = (slide_img * mask).astype(np.uint8)
        split_slide(masked_slide_img, slide_img_name, save_dir, stride = stride, save_black = False)

def excel_concat(excel_path_list, save_path):
    for i , excel_path in enumerate(excel_path_list):
        if i == 0:
            df = pd.read_excel(excel_path, index_col=0).dropna()
        else:
            df_temp = pd.read_excel(excel_path, index_col=0).dropna()
            df = pd.concat([df,df_temp], ignore_index=True)
    df.to_excel(save_path)

def train_val_split(excel_path, val_ratio = 0.2):
    df = pd.read_excel(excel_path, index_col=0)
    train_slides = []
    val_slides = []
    for slide_name in set(df['original_slide_name']):
        if np.random.rand() > val_ratio:
            train_slides.append(slide_name)
        else:
            val_slides.append(slide_name)
    print('training slides: ')
    print(train_slides)
    print('validation slides')
    print(val_slides)
    
    if input('Is this split OK? (y for yes) ') == 'y':
        for index, item in df.iterrows():
            if item['original_slide_name'] in train_slides:
                df.loc[index, 'mode'] = 'train'
            else:
                df.loc[index, 'mode'] = 'val'
        df.to_excel(excel_path)
        print()
        print('excel saved:', excel_path)
    else:
        print()
        print('run this cell again to generate new split')

def generate_subDataset(subDataset_dir, excel_path, classes, non_img_generation = [], 
                        num_random_crop = 1, label_col = 'class', stride = 640):
    edp = ExcelDataProcessor(excel_path, h=stride, w=stride)

    refresh_directory(subDataset_dir)

    # generate training dataset
    mode = 'train'
    save_dir = os.path.join(subDataset_dir, 'train')
    conditions = {'class': classes, 'mode': ['train']}
    edp.create(save_dir, conditions, label_col=label_col, num_random_crop=num_random_crop,
               mode=mode, stride=stride, non_img_generation = non_img_generation)

    # generate validation dataset
    mode = 'train' # mode for edp.create should also be 'train' (sorry for confusing...)
    save_dir = os.path.join(subDataset_dir, 'val')
    conditions = {'class': classes, 'mode': ['val']}
    edp.create(save_dir, conditions, label_col=label_col, num_random_crop=num_random_crop,
               mode=mode, stride=stride, non_img_generation = non_img_generation)



def generate_subDataset_onlyImg(ROOT_DIR, sample_ID, subDataset_dir, excel_path, stride = 640):
    df = pd.read_excel(excel_path, index_col=0)
    print(len(df))
    slide_img_dir = os.path.dirname(glob(f'{ROOT_DIR}data/images/*/*/{sample_ID}*.jpg')[0])
    refresh_directory(subDataset_dir)

    for trainVal in ['train', 'val']:
        json_data = {}
        save_dir = os.path.join(subDataset_dir, trainVal)
        os.mkdir(save_dir)
        df_trainVal = df[df['mode'] == trainVal]
        original_slide_name_list = sorted(set(df_trainVal['original_slide_name']))

        for original_slide_name in original_slide_name_list:
            original_slide_path = os.path.join(slide_img_dir, original_slide_name)
            original_slide_img = cv2.imread(original_slide_path)
            df_img = df_trainVal[df_trainVal['original_slide_name'] == original_slide_name]
            approx_strat_points_hw = []

            for index, item in df_img.iterrows():
                x_min, y_min = int(item['x_min']), int(item['y_min'])
                x_max, y_max = int(item['x_max']), int(item['y_max'])
                mode = item['mode']

                start_h = random.randrange(max(0, y_max - stride), max(1, min(y_min, original_slide_img.shape[0] - stride)), 1)
                start_w = random.randrange(max(0, x_max - stride), max(1, min(x_min, original_slide_img.shape[1] - stride)), 1)

                approx_start_hw = (int(round(start_h, -2)), int(round(start_w, -2)))

                if approx_start_hw not in approx_strat_points_hw:
                    approx_strat_points_hw.append(approx_start_hw)
                    new_img = original_slide_img[start_h:start_h+stride, start_w:start_w + stride]
                    save_name = '{}_{}_{}.jpg'.format(original_slide_name[:-4], start_h, start_w)
                    save_path = os.path.join(save_dir, save_name)
                    cv2.imwrite(save_path, new_img)

                    json_data = generate_json_template(json_data, save_path)

        if len(json_data.keys()) >= 1:
            json_path = os.path.join(save_dir, f'{sample_ID}_{trainVal}.json')
            save_json(json_data, json_path)
        else:
            print(f'No data: {sample_ID}_{trainVal}')



def integrate_subDataset(subDataset_path_list, not_for_dataset, dataset_name, class_dict):
    subDataset_path_list = [sdp for sdp in subDataset_path_list if sdp not in not_for_dataset]
    for sdp in subDataset_path_list: # sdp: subDataset_path
        print(sdp)
    assert input('generating dataset by these sub-dataset ? (y for yes) ') == 'y', \
        'modify not_for_dataset or subDataset_path_list'

    dataset_path = os.path.join(f'../data/dataset/{dataset_name}')
    refresh_directory(dataset_path)
    
    for mode in ['train', 'val']:
        dataset_trainVal = os.path.join(dataset_path, mode)
        os.mkdir(dataset_trainVal)
        json_data = {}

        for sdp in subDataset_path_list:
            sdp_trainVal = os.path.join(sdp, mode)
            sub_json_path = os.path.join(sdp_trainVal, 'via_region_data.json')
            
            if not os.path.exists(sub_json_path):
                print('No json file:', sub_json_path)
            else:
                sub_json_data = load_json(sub_json_path)

                # copy images
                img_path_list = glob(os.path.join(sdp_trainVal, '*.jpg'))
                for img_path in img_path_list:
                    img_name = os.path.basename(img_path)
                    new_path = os.path.join(dataset_trainVal, img_name)
                    shutil.copy(img_path, new_path)

                # integrate json data
                for k1, v1 in sub_json_data.items():
                    json_data[k1] = v1

        json_path = os.path.join(dataset_trainVal, 'via_region_data.json')
        save_json(json_data, json_path)
        
        convert_class_label(json_path, class_dict, json_path)



def check_dataset(dataset_name, remove_classes = []):
    df = pd.DataFrame()
    dataset_dir = f'../data/dataset/{dataset_name}'

    index = 1
    for subset in ['train', 'val']:
        class_count = {}
        subset_dir = os.path.join(dataset_dir, subset)
        json_path = os.path.join(subset_dir, 'via_region_data.json')
        json_data = load_json(json_path)
        df.loc[index, 'subset'] = subset
        df.loc[index, 'num_img'] = len(json_data)
        for v1 in json_data.values():
            for v2 in v1['regions'].values():
                class_name = v2['region_attributes']['name']

                # 特定のクラスの画像を削除
                if class_name in remove_classes:
                    img_path = os.path.join(subset_dir, v1['filename'])
                    try:
                        os.remove(img_path)
                    except:
                        print('skip:', v1['filename'])

                else:
                    if class_name in class_count.keys():
                        class_count[class_name] += 1
                    else:
                        class_count[class_name] = 1
                    
        for cl, count in class_count.items():
            df.loc[index, cl] = count
        index += 1
        
        if subset == 'train':
            df.loc[3, 'subset'] = '(class_weight)'
            for cl in class_count.keys():
                weight = np.sum(df.loc[1, class_count.keys()]) / df.loc[1, cl] / len(class_count)
                df.loc[3, cl] = round(weight, 3)
    df = df.sort_index()
    print(df)
    excel_path = os.path.join(dataset_dir, 'count.xlsx')
    df.to_excel(excel_path)


        
# process_excel のコピー，本当は呼び出したい
def _get_size(all_points_x, all_points_y, pixel2micron = 2.54):
    contour = np.array([all_points_x, all_points_y], dtype=np.int32).T
    rect = cv2.minAreaRect(contour)
    size = rect[1][0] * rect[1][1] * (pixel2micron ** 2)
    return size

# 座標の最大・最小から面積を取得 (load_from_json, load_from_pickle)
def _get_length(all_points_x, all_points_y, pixel2micron = 2.54):
    contour = np.array([all_points_x, all_points_y], dtype=np.int32).T
    rect = cv2.minAreaRect(contour)
    length = int(round(np.max(rect[1]) * pixel2micron))
    return length

def _to_string(points_array):
    points_array = points_array.astype(int)
    points_array = map(str, points_array.tolist())
    str_points_array = ', '.join(points_array)
    return str_points_array

    
def refresh_directory(directory):
    print('refresh directory: ', directory)
    if os.path.exists(directory):
        if input('Are you sure to delete the directory (y for yes) ? ') == 'y':
            shutil.rmtree(directory)
        else:
            assert False, 'please check directory path'
    os.makedirs(directory)

