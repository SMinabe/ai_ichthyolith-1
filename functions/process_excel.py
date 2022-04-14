import os
import sys
import json
import math
import random
import pickle
import shutil
from glob import glob
import xml.etree.ElementTree as ET
from skimage.io import imread, imsave

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

from functions.utils import save_pickle, save_json, predict_contour
from functions.process_json import format_to_json

random.seed(0)
np.random.seed(0)


class ExcelDataProcessor:

    def __init__(self, excel_path, h=640, w=640):
        self.excel_path = excel_path
        self.columns = self.get_columns()
        self.slides_path = glob('../data/images/*/*/*.jpg')
        self.h = h
        self.w = w
        self.mask_annotations = dict()
        self.new_dataset_df = None

    def get_columns(self):
        """
        列名を取得する
        :return: 列名のリスト (list)
        """
        df = self.read_excel()
        columns = list(df.columns)
        print('columns: ', columns)
        return columns

    def add_columns(self, col_name):
        """
        列名を追加する
        :param col_name:
        :return:
        """
        self.columns.append(col_name)

    def read_excel(self):
        """
        excelデータの読み込む
        :return: データフレーム (pd.DataFrame)
        """
        return pd.read_excel(self.excel_path, header=0, index_col=0)

    def get_identical_slide_path(self, original_slide_name):
        """
        original_slide_nameに含まれるスライド画像のパスを取得する
        :param original_slide_name:
        :return:
        """
        for path in self.slides_path:
            if path.endswith(original_slide_name):
                return path
        return None

    def load_from_json(self, json_path, sheet_name=None):
        """
        viaで作成したjsonファイルからアノテーションデータを読み込む
        :param json_path:
        :param sheet_name:
        :return:
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = self.read_excel()

        if len(df.index) != 0:
            index = df.index[-1] + 1
        else:
            index = 1
        if index == 1:
            identical_number = 1
        else:
            identical_number = int(df['id'].max()) + 1
        for v in data.values():
            original_slide_name = self._get_slide_name(v['filename'])
            start_x, start_y = self._get_start_coordinates(v['filename'])
            regions = v['regions']
            for region in regions.values():
                all_points_x = np.array(region['shape_attributes']['all_points_x']) + start_x
                all_points_y = np.array(region['shape_attributes']['all_points_y']) + start_y
                x_min, y_min, x_max, y_max = self._get_box_coordinates(all_points_x, all_points_y)
                size = self._get_size(all_points_x, all_points_y)
                length = self._get_length(all_points_x, all_points_y)
                str_all_points_x = self._to_string(all_points_x)
                str_all_points_y = self._to_string(all_points_y)
                class_name = region['region_attributes']['name'] if 'name' in region['region_attributes'] else ''
                new_row = self._create_new_row(identical_number, original_slide_name, x_min, y_min, x_max, y_max,
                                               str_all_points_x, str_all_points_y, size, length, class_name, index)
                df = df.append(new_row, sort=False)
                identical_number += 1
                index += 1
        df = df[self.columns]
        if sheet_name is not None:
            df.to_excel(self.excel_path, index=list(df.index), header=df.columns, sheet_name=sheet_name)
        else:
            df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('finished loading from json.')

    def load_from_pickle(self, pickle_path, labels, sheet_name=None):
        """
        ichthyolith_test.pyによって作成されたpickleファイルから座標データを読み込む
        :param pickle_path:
        :param labels:
        :param sheet_name:
        :return:
        """
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        df = self.read_excel()

        if len(df.index) != 0:
            index = df.index[-1] + 1
        else:
            index = 1
        if index == 1:
            identical_number = 1
        else:
            identical_number = int(df['id'].max()) + 1
        for d in data:
            original_slide_name = self._get_slide_name(d[0]['filename'])
            start_x, start_y = self._get_start_coordinates(d[0]['filename'])
            class_ids = d[0]['class_ids']
            class_names = [labels[i] for i in class_ids.tolist()]
            for i in range(len(d[0]['rois'])):
                all_points_x, all_points_y = self._get_all_points_xy_from_mask(d[0]['masks'][:, :, i])
                all_points_x = all_points_x + start_x
                all_points_y = all_points_y + start_y
                x_min, y_min, x_max, y_max = self._get_box_coordinates(all_points_x, all_points_y)
                size = self._get_size(all_points_x, all_points_y)
                length = self._get_length(all_points_x, all_points_y)
                str_all_points_x = self._to_string(all_points_x)
                str_all_points_y = self._to_string(all_points_y)
                class_name = class_names[i]
                new_row = self._create_new_row(identical_number, original_slide_name, x_min, y_min, x_max, y_max,
                                               str_all_points_x, str_all_points_y, size, length, class_name, index)
                df = df.append(new_row, sort=False)
                identical_number += 1
                index += 1
        df = df[self.columns]
        if sheet_name is not None:
            df.to_excel(self.excel_path, index=list(df.index), header=df.columns, sheet_name=sheet_name)
        else:
            df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('finished loading from pickle.')

    
    # todo: 機能分割
    def load_from_xml(self, ROOT_DIR, xml_dir, trimmed_class_dir = '', sheet_name=None, 
                      margin = None, slide_img_dir = None, cl_name = None):
        """
        labelImg で作成した xml ファイルからアノテーションデータを読み込む（見邨）
        :param xml_dir:
        :param save_excel_path:
        :param sheet_name:
        :return:
        """
        # format excel
        df = self.read_excel()

        if len(df.index) != 0:
            index = df.index[-1] + 1
        else:
            index = 1
        if index == 1:
            identical_number = 1
        else:
            identical_number = int(df['id'].max()) + 1

        json_data = {}
        
        previous_slide_name = ''
        xml_path_list = glob(os.path.join(xml_dir, ('*.xml')))
        for xml_path in xml_path_list:
            xml_file = os.path.basename(xml_path)
            split_img_name = f'{xml_file[:-4]}.jpg'
            # slide_img_name = f'{xml_file[:-10]}.jpg'
            slide_img_name = '_'.join(xml_file.split('_')[:-2]) + '.jpg'

            if slide_img_dir is None:
                slide_img_dir = os.path.dirname(glob(f'{ROOT_DIR}data/images/*/*/{slide_img_name}')[0])
            slide_img_path = os.path.join(slide_img_dir, slide_img_name)
            if slide_img_name != previous_slide_name:
                slide_img = cv2.imread(slide_img_path)
                assert slide_img is not None, f'path not exist: {slide_img_path}'
                previous_slide_name = slide_img_name
            # i, j = int(split_img_name[:-4].split("_")[-2]), int(split_img_name[:-4].split("_")[-1])
            y, x = int(split_img_name[:-4].split("_")[-2]), int(split_img_name[:-4].split("_")[-1])
            
            # load xml file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for Object in [child for child in root if child.tag == "object"]:
                # get class label
                class_name = Object[0].text
                    
                # get position in split image
                x_min = int(Object[4][0].text)
                y_min = int(Object[4][1].text)
                x_max = int(Object[4][2].text)
                y_max = int(Object[4][3].text)

                # add margin and convert to position in slide image
                if margin is None:
                    margin = max(50, int((x_max - x_min) * 0.3), int((y_max - y_min) * 0.3))
                X_min = max(x_min + x - margin, 0)
                Y_min = max(y_min + y - margin, 0)
                X_max = min(x_max + x + margin, slide_img.shape[1])
                Y_max = min(y_max + y + margin, slide_img.shape[0])                
                
                # get incorrect all_points_x and all_points_y (just for formatting)
                all_points_x = [X_min, X_max, X_max, X_min, X_min]
                all_points_y = [Y_min, Y_min, Y_max, Y_max, Y_min]
                str_all_points_x = (', ').join(map(str, all_points_x))
                str_all_points_y = (', ').join(map(str, all_points_y))
                size = (X_max - X_min) * (Y_max - Y_min)
                length = min((X_max - X_min), (Y_max - Y_min))


                # write image and predict contour
                if os.path.exists(trimmed_class_dir):
                    trimmed_img = slide_img[Y_min:Y_max, X_min:X_max]
                    trimmed_img_name = '{:0=4}_{}'.format(index, slide_img_name)
                    trimmed_img_path = os.path.join(trimmed_class_dir, trimmed_img_name)
                    cv2.imwrite(trimmed_img_path, trimmed_img)

                    contour = predict_contour(trimmed_img)
                    json_data = format_to_json(json_data, trimmed_img_path, [contour], class_name)


                elif trimmed_class_dir != '':
                    print(f'ERROR, path not exists: {trimmed_class_dir}')
                    print('skip writing trimmed images')

                # write in excel file
                new_row = self._create_new_row(identical_number, slide_img_name, X_min, Y_min, X_max, Y_max,
                                            str_all_points_x, str_all_points_y, size, length, class_name, index)
                df = df.append(new_row, sort=False)
                index += 1
                identical_number += 1
        
        # save excel file    
        if sheet_name is not None:
            df.to_excel(self.excel_path, index=list(df.index), header=df.columns, sheet_name=sheet_name)
        else:
            df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('saved', self.excel_path)

        # save json file
        if os.path.exists(trimmed_class_dir):
            json_path = os.path.join(trimmed_class_dir, 'region_data_unchecked.json')
            save_json(json_data, json_path) 
        
    
    
    
    
    
    
    
    
    def put_labels(self, labels, col_name, conditions=None, display_mode='box'):
        """
        クラスラベルを付与する
        :param labels: 付与するラベルのリスト (list)
        :param col_name: ラベルを付ける列名 (str)
        :param conditions: チェックするレコードの条件 (dict)
        None: col_nameが空のレコード
        ex) {'class': ['tooth', 'denticle']} : 'classの列がtoothまたはdenticleのレコード
        :param display_mode: box or mask
        :return:
        """
        labels.insert(0, 'recheck')
        df = self.read_excel()
        # df = pd.read_excel(self.excel_path, header=0, index_col=0)
        if col_name not in self.columns:
            self.add_columns(col_name)
            df[col_name] = None
        df[col_name] = df[col_name].astype(object)
        sub_df = self.get_sub_df(df, conditions, col_name)
        index = list(sub_df.index)
        pointer = 0
        h, w = self.h, self.w
        last_image_path = ''
        last_img = ''
        while pointer < len(sub_df):
            row = sub_df.loc[index[pointer]]
            img_path = self.get_identical_slide_path(row['original_slide_name'])
            if last_image_path == '' or img_path != last_image_path:
                img = cv2.imread(img_path)
                last_img = img
            else:
                img = last_img
            x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            start_h = max(0, int(y_center - (h / 2)))
            start_w = max(0, int(x_center - (w / 2)))
            display_img = img[start_h:start_h+h, start_w:start_w+w]
            window_name = os.path.basename(img_path) + f'({start_h}, {start_w}), index: {index[pointer]}'
            if display_mode == 'box':
                top_left = (int(max(0, x_min-start_w-5)), int(max(0, y_min-start_h-5)))
                bottom_right = (int(min(w, x_max-start_w+5)), int(min(h, y_max-start_h+5)))
                display_img = cv2.rectangle(display_img, top_left, bottom_right, (0, 0, 255), thickness=2)
                cv2.imshow(window_name, display_img)
            else:
                all_points_x = np.array(list(map(int, row['all_points_x'].split(', '))))
                all_points_y = np.array(list(map(int, row['all_points_y'].split(', '))))
                all_points_x = all_points_x - start_w
                all_points_y = all_points_y - start_h
                pts = self.convert_to_vertex(all_points_x, all_points_y)
                display_img = cv2.polylines(display_img, pts, True, (0, 0, 255), thickness=1)
                cv2.imshow(window_name, display_img)
            # print(f'現在のラベル: {df.at[index[pointer], col_name]}')
            sys.stdout.write(f'\r現在のラベル: {df.at[index[pointer], col_name]}\n')
            self.display_operation(labels)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('b'):
                pointer -= 1
                cv2.destroyAllWindows()
            elif key == ord('n'):
                pointer += 1
                cv2.destroyAllWindows()
            elif int(chr(key)) < len(labels):
                df.at[index[pointer], col_name] = labels[int(chr(key))]
                pointer += 1
                cv2.destroyAllWindows()
            else:
                cv2.destroyAllWindows()
        df = df[self.columns]
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('saved.')

    # TODO: clean up code to reduce duplicated parts
    def create(self, save_dir, conditions, label_col='class', num_random_crop=1, 
               mode='train', stride=640, non_img_generation = []):
        """
        学習データの作成
        :param save_dir:
        :param conditions:
        :param label_col:
        :param num_random_crop:
        :param mode:
        :param stride:
        :return:
        """
        os.makedirs(save_dir, exist_ok=True)
        df = self.read_excel()
        for col_name, labels in conditions.items():
            df = df[df[col_name].isin(labels)]
        original_slide_list = sorted(list(set(df['original_slide_name'])))
        h, w = self.h, self.w
        print(df.index)
        for i, original_slide_name in enumerate(original_slide_list):
            slide_path = self.get_identical_slide_path(original_slide_name)
            slide = cv2.imread(slide_path)
            sub_df = df[df['original_slide_name'] == original_slide_name]
            
            if mode == 'test':
                slide_h, slide_w, _ = slide.shape
                iter_h = math.floor(slide_h / h)
                iter_w = math.floor(slide_w / w)
                for ih in range(iter_h):
                    for jw in range(iter_w):
                        start_h = ih * h
                        start_w = jw * w
                        img = slide[start_h:start_h + h, start_w:start_w + w]
                        df_contained = self._get_df_contained(sub_df, start_h, start_w)
                        if len(df_contained) == 0:
                            continue
                        annotations = []
                        for index2 in list(df_contained.index):
                            try:
                                mask = self._get_mask(df_contained, index2, label_col, start_h, start_w)
                                self._process_new_dataset_df(df_contained, index2, label_col,
                                                             original_slide_name, start_h, start_w)
                                annotations.append(mask)
                            except Exception as e:
                                print(f'index: {index2}')
                                print(f'exception: {e}')
                                continue
                        base = original_slide_name.split('.')[0]
                        save_img_name = f'{base}_{str(start_h).zfill(2)}_{str(start_w).zfill(2)}.jpg'
                        save_img_path = os.path.join(save_dir, save_img_name)
                        cv2.imwrite(save_img_path, img)
                        print(f'saved: {save_img_name}')
                        self._process_annotations_for_mask_rcnn(annotations, save_img_path)

            elif mode == 'annot':
                slide_h, slide_w, _ = slide.shape
                iter_h = math.floor((slide_h - h + stride) / stride)
                iter_w = math.floor((slide_w - w + stride) / stride)
                for ih in range(iter_h):
                    for jw in range(iter_w):
                        start_h = ih * stride
                        start_w = jw * stride
                        img = slide[start_h:start_h+h, start_w:start_w+w]
                        df_contained = self._get_df_contained(sub_df, start_h, start_w)
                        annotations = []
                        if len(df_contained) != 0:
                            for index2 in list(df_contained.index):
                                try:
                                    mask = self._get_mask(df_contained, index2, label_col, start_h, start_w)
                                    self._process_new_dataset_df(df_contained, index2, label_col,
                                                                 original_slide_name, start_h, start_w)
                                    annotations.append(mask)
                                except Exception as e:
                                    print(f'index: {index2}')
                                    print(f'exception: {e}')
                                    continue
                        base = original_slide_name.split('.')[0]
                        save_img_name = f'{base}_{str(start_h).zfill(2)}_{str(start_w).zfill(2)}.jpg'
                        save_img_path = os.path.join(save_dir, save_img_name)
                        cv2.imwrite(save_img_path, img)
                        print(f'saved: {save_img_name}')
                        if len(annotations) != 0:
                            self._process_annotations_for_mask_rcnn(annotations, save_img_path)
            else: # mode == "train"
                for index in list(sub_df.index):

                    # 20210215_見邨追加
                    if sub_df.loc[index, 'class'] not in non_img_generation:
                        
                        x_min = sub_df.loc[index, 'x_min']
                        y_min = sub_df.loc[index, 'y_min']
                        x_max = sub_df.loc[index, 'x_max']
                        y_max = sub_df.loc[index, 'y_max']
                        for _ in range(num_random_crop):
                            try:
                                if mode == 'train':
                                    start_h = random.randrange(max(0, y_max - h), max(1, min(y_min, slide.shape[0] - h)), 1)
                                    start_w = random.randrange(max(0, x_max - w), max(1, min(x_min, slide.shape[1] - w)), 1)
                                else:
                                    start_h = int(max(0, (y_min+y_max)/2-(h/2)))
                                    start_w = int(max(0, (x_min+x_max)/2-(w/2)))
                            except Exception as e:
                                print(f'index: {index}')
                                print(f'exception: {e}')
                                continue
                            img = slide[start_h:start_h+h, start_w:start_w+w]
                            df_contained = self._get_df_contained(sub_df, start_h, start_w)
                            annotations = []
                            for index2 in list(df_contained.index):
                                try:
                                    mask = self._get_mask(df_contained, index2, label_col, start_h, start_w)
                                    self._process_new_dataset_df(df_contained, index2, label_col,
                                                                original_slide_name, start_h, start_w)
                                    annotations.append(mask)
                                except Exception as e:
                                    print(f'index: {index2}')
                                    print(f'exception: {e}')
                                    continue
                            base = original_slide_name.split('.')[0]
                            save_img_name = f'{base}_{str(start_h).zfill(2)}_{str(start_w).zfill(2)}.jpg'
                            save_img_path = os.path.join(save_dir, save_img_name)
                            cv2.imwrite(save_img_path, img)
                            print(f'saved: {save_img_name}')
                            self._process_annotations_for_mask_rcnn(annotations, save_img_path)
        self._save_annotations_for_mask_rcnn(save_dir)
        self._save_new_dataset_df(save_dir)

    # TODO: cleanup
    def create_eval_gts(self, save_path, conditions, class_names):
        df = self.read_excel()
        sub_df = self.get_sub_df(df, conditions, None)
        gts = []
        filenames = list(set([df.loc[i, 'original_slide_name'] for i in list(sub_df.index)]))
        filenames.sort()
        for filename in filenames:
            print(filename)
            temp_df = sub_df[sub_df['original_slide_name'] == filename]
            gt_bbox = []
            # TODO: メモリオーバー改善
            gt_mask = np.zeros((1, 1, len(temp_df)), dtype=np.int8)
            gt_class_id = np.array([], dtype=np.int8)
            gt_size = np.array([], dtype=float)
            gt_length = np.array([], dtype = np.int16)
            for index in list(temp_df.index):
                gt_bbox.append([temp_df.loc[index, 'y_min'], temp_df.loc[index, 'x_min'],
                                temp_df.loc[index, 'y_max'], temp_df.loc[index, 'x_max']])
                gt_class_id = np.append(gt_class_id, class_names.index(temp_df.loc[index, 'class']))
                # TODO: gt_maskを作成する機能 --> mask は作成していないが size と length は取れたので大体解決
                record_size = False
                if pd.notna(temp_df.loc[index, 'size']):
                    gt_size = np.append(gt_size, temp_df.loc[index, 'size'])
                    gt_length = np.append(gt_length, temp_df.loc[index, 'length'])
                    record_size = True
            gt_bbox = np.array(gt_bbox, dtype=np.int16)
            contents = {"filename": filename, "gt_bbox": gt_bbox.astype("int16"),
                        "gt_class_id": gt_class_id.astype("int8"), "gt_mask": gt_mask.astype("int8")}
            if record_size:
                contents['gt_size'] = gt_size
                contents['gt_length'] = gt_length

            gts.append([contents])
        save_pickle(gts, save_path)

    def display(self, conditions=None):
        pass

    def save_min_max(self, conditions=None):
        """
        輪郭座標から最小・最大を計算して保存
        :param conditions:
        :return:
        """
        df = self.read_excel()
        index = [i + 1 for i in range(len(df))]
        df.index = index
        col_name = 'x_min'
        sub_df = self.get_sub_df(df, conditions, col_name)
        for index in list(sub_df.index):
            try:
                all_points_x = np.array(list(map(int, sub_df.loc[index, 'all_points_x'].split(', '))))
                all_points_y = np.array(list(map(int, sub_df.loc[index, 'all_points_y'].split(', '))))
                x_min, y_min, x_max, y_max = self._get_box_coordinates(all_points_x, all_points_y)
                df.at[index, 'x_min'] = x_min
                df.at[index, 'y_min'] = y_min
                df.at[index, 'x_max'] = x_max
                df.at[index, 'y_max'] = y_max
            except Exception as e:
                print(f'index: {index}')
                print(e)
        df = df[self.columns]
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('saved x_min, y_min, x_max, y_max.')

    def save_size(self, conditions=None):
        """
        マスクの面積を計算して保存
        :param conditions:
        :return:
        """
        df = self.read_excel()
        index = [i + 1 for i in range(len(df))]
        df.index = index
        col_name = 'size'
        sub_df = self.get_sub_df(df, conditions, col_name)
        for index in list(sub_df.index):
            all_points_x = np.array(list(map(int, sub_df.loc[index, 'all_points_x'].split(', '))))
            all_points_y = np.array(list(map(int, sub_df.loc[index, 'all_points_y'].split(', '))))
            size = self._get_size(all_points_x, all_points_y)
            df.at[index, 'size'] = size
        df = df[self.columns]
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('saved size.')

    def save_length(self, conditions=None):
        df = self.read_excel()
        index = [i + 1 for i in range(len(df))]
        df.index = index
        col_name = 'length'
        sub_df = self.get_sub_df(df, conditions, col_name)
        for index in list(sub_df.index):
            all_points_x = np.array(list(map(int, sub_df.loc[index, 'all_points_x'].split(', '))))
            all_points_y = np.array(list(map(int, sub_df.loc[index, 'all_points_y'].split(', '))))
            size = self._get_length(all_points_x, all_points_y)
            df.at[index, 'length'] = size
        df = df[self.columns]
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('saved length.')

    def sort(self, col_names, ascending=None):
        """
        col_namesを基準に並び替えさせる
        :param col_names:
        :param ascending:
        :return:
        """
        df = self.read_excel()
        if ascending:
            df = df.sort_values(col_names, ascending=ascending)
        else:
            df = df.sort_values(col_names)
        index = [i + 1 for i in range(len(df))]
        df.index = index
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print(f'sorted by {col_names}, ascending={ascending}')

    def batch_update(self, conditions, col_name, value):
        """
        条件を満たすイクチオリスのcol_nameを一括で変更する
        :param conditions:
        :param col_name:
        :param value:
        :return:
        """
        df = self.read_excel()
        index = [i + 1 for i in range(len(df))]
        df.index = index
        sub_df = self.get_sub_df(df, conditions, col_name)
        for index in list(sub_df.index):
            df.at[index, col_name] = value
        df = df[self.columns]
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        print('conditions:')
        for col, val in conditions.items():
            print(f'col_name: {col} = {val}')
        print()
        print(f'update {col_name}. value: {value}')

    def describe(self, col_name, conditions, include=True):
        """
        各種統計量の表示
        :param col_name:
        :param conditions:
        :param include:
        :return:
        """
        df = self.read_excel()
        sub_df = self.get_sub_df(df, conditions, col_name, include=include)
        print(f'column: {col_name}')
        print(sub_df[col_name].describe())

    def hist(self, col_name, conditions, bins=100, x_range=None, normed=False, include=True):
        """
        ヒストグラムの表示
        :param col_name:
        :param conditions:
        :param bins:
        :param x_range:
        :param normed:
        :param include:
        :return:
        """
        df = self.read_excel()
        sub_df = self.get_sub_df(df, conditions, col_name, include=include)
        x = sub_df[col_name].values
        if normed:
            weights = np.ones(len(x)) / float(len(x))
        else:
            weights = np.ones(len(x))
        if x_range is None:
            plt.hist(x, bins=bins, label=col_name, weights=weights)
        else:
            plt.hist(x, bins=bins, label=col_name, range=x_range, weights=weights)
        plt.title(f'{col_name}: conditions={conditions.keys()}, include={include}')
        plt.legend(loc='upper right')
        plt.show()

    def replace_labels(self, labels_dict, col_name):
        """
        ラベルを置換する
        :param labels_dict:
        :param col_name:
        :return:
        """
        df = self.read_excel()
        df = df.replace({col_name: labels_dict})
        df = df[self.columns]
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)
        for pre_label, post_label in labels_dict.items():
            print(f'replaced {pre_label} with {post_label} in \'{col_name}\' column')

    def save_sub_dataset(self, save_path, conditions):
        """
        特定のラベルのデータを抽出し，別ファイルに保存
        :param save_path:
        :param conditions:
        :return:
        """
        df = self.read_excel()
        sub_df = self.get_sub_df(df, conditions, '')
        sub_df = sub_df[self.columns]
        sub_df.to_excel(save_path, index=list(sub_df.index), header=sub_df.columns)
        print(f'saved sub dataset: {save_path}')

    def delete_data(self, labels, col_name):
        """
        特定のラベルのデータの削除
        :param labels:
        :param col_name:
        :return:
        """
        df = self.read_excel()
        df = df[~df[col_name].isin(labels)]
        index = [i + 1 for i in range(len(df))]
        df.index = index
        df = df[self.columns]
        df.to_excel(self.excel_path, index=list(df.index), header=df.columns)

    def get_ids(self, labels, col_name):
        """
        特定のラベルのidを取得
        :param labels:
        :param col_name:
        :return:
        """
        df = self.read_excel()
        ids = list(df[df[col_name].isin(labels)].index)
        return ids

    def show_col_names(self):
        """
        エクセルデータの列名を表示
        :return:
        """
        print(self.columns)

    def backup(self, backup_path):
        """
        エクセルデータのバックアップを作成
        :param backup_path:
        :return:
        """
        shutil.copyfile(self.excel_path, backup_path)
        print('made backup file.')
        print(f'file path: {backup_path}')

    def reset(self):
        """エクセルデータを初期化"""
        df = pd.DataFrame(columns=self.columns)
        df.to_excel(self.excel_path, header=df.columns)
        print('reset dataset!')

    def save_min_max_size_length(self):
        self.save_min_max()
        self.save_size()
        self.save_length()

    def reset_mask_annotations(self):
        self.mask_annotations = dict()

    def _get_df_contained(self, df, start_h, start_w):
        df_x_min_ok = df[df['x_min'] > start_w]
        df_x_ok = df_x_min_ok[df_x_min_ok['x_max'] < start_w+self.w]
        df_x_y_min_ok = df_x_ok[df_x_ok['y_min'] > start_h]
        df_contained = df_x_y_min_ok[df_x_y_min_ok['y_max'] < start_h+self.h]
        return df_contained

    def _process_annotations_for_mask_rcnn(self, annotations, img_path):
        img_size = os.path.getsize(img_path)
        key = os.path.basename(img_path) + str(img_size)
        regions = dict()
        for i, mask in enumerate(annotations):
            shape_attributes = self._make_shape_attributes(mask)
            region_attributes = self._make_region_attributes(mask)
            regions[str(i)] = {"shape_attributes": shape_attributes, "region_attributes": region_attributes}
        value = {"fileref": "", "size": img_size, "filename": os.path.basename(img_path),
                 "base64_img_data": "", "file_attributes": {}, "regions": regions}
        self.mask_annotations[key] = value

    def _save_annotations_for_mask_rcnn(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'via_region_data.json')
        with open(save_path, 'w') as f:
            json.dump(self.mask_annotations, f)
        self.reset_mask_annotations()

    def _process_new_dataset_df(self, df_contained, index, label_col, original_slide_name, start_h, start_w):
        columns = ['id', 'image_name', 'original_slide_name',
                   'all_points_x', 'all_points_y',
                   'size', 'length', 'label']
        if self.new_dataset_df is None:
            self.new_dataset_df = pd.DataFrame(columns=columns)
        identical_number = df_contained.loc[index, 'id']
        base = original_slide_name.split('.')[0]
        image_name = f'{base}_{str(start_h).zfill(2)}_{str(start_w).zfill(2)}.jpg'
        all_points_x = np.array(list(map(int, df_contained.loc[index, 'all_points_x'].split(', ')))) - start_w
        all_points_y = np.array(list(map(int, df_contained.loc[index, 'all_points_y'].split(', ')))) - start_h
        str_all_points_x = self._to_string(all_points_x)
        str_all_points_y = self._to_string(all_points_y)
        size = df_contained.loc[index, 'size']
        length = df_contained.loc[index, 'length']
        label = df_contained.loc[index, label_col]
        new_row = pd.DataFrame([[identical_number, image_name, original_slide_name,
                                 str_all_points_x, str_all_points_y,
                                 size, length, label]],
                               columns=columns)
        self.new_dataset_df = self.new_dataset_df.append(new_row, sort=False)

    def _save_new_dataset_df(self, save_dir):
        save_path = os.path.join(save_dir, 'ichthyolith_info.xlsx')

        # # 見邨追記
        # if self.new_dataset_df is not None:

        index = [i + 1 for i in range(len(self.new_dataset_df))]
        columns = ['id', 'image_name', 'original_slide_name',
                'all_points_x', 'all_points_y',
                'size', 'length', 'label']
        self.new_dataset_df.index = index
        self.new_dataset_df.to_excel(save_path, index=index, header=columns)
        self.new_dataset_df = None
        print('saved.')

    @staticmethod
    def _get_slide_name(filename):
        """
        画像名から元スライド名を取得する　(load_from_json, load_from_pickle)
        :param filename: 画像名
        :return original_slide_name: 元スライド名
        """
        parts = filename.split('_')
        original_slide_name = '_'.join(parts[:-2]) + '.jpg'
        return original_slide_name

    @staticmethod
    def _get_start_coordinates(filename):
        """
        画像名から左上の座標を取得する (load_from_json, load_from_pickle)
        :param filename: 画像名
        :return start_x, start_y: 画像の左上の座標
        """
        parts = filename.split('_')
        start_y = np.array(int(parts[-2]))
        start_x = np.array(int(parts[-1].replace('.jpg', '')))
        return start_x, start_y

    @staticmethod
    def _get_box_coordinates(all_points_x, all_points_y):
        if not len(all_points_x) == 0 and not len(all_points_y) == 0:
            x_min = int(all_points_x.min())
            y_min = int(all_points_y.min())
            x_max = int(all_points_x.max())
            y_max = int(all_points_y.max())
        else:
            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _get_size(all_points_x, all_points_y):
        contour = np.array([all_points_x, all_points_y], dtype=np.int32).T
        rect = cv2.minAreaRect(contour)
        size = rect[1][0] * rect[1][1]
        return size

    @staticmethod
    # 座標の最大・最小から面積を取得 (load_from_json, load_from_pickle)
    def _get_length(all_points_x, all_points_y, pixel2micron = 2.54):
        contour = np.array([all_points_x, all_points_y], dtype=np.int32).T
        rect = cv2.minAreaRect(contour)
        length = round(np.max(rect[1]) * pixel2micron)
        return length

    @staticmethod
    def _to_string(points_array):
        points_array = points_array.astype(int) # 見邨追記
        points_array = map(str, points_array.tolist())
        str_points_array = ', '.join(points_array)
        return str_points_array

    @staticmethod
    def _create_new_row(identical_number, original_slide_name, x_min, y_min, x_max, y_max,
                        str_all_points_x, str_all_points_y, size, length, class_name, index):
        new_row = pd.DataFrame([[identical_number, original_slide_name, x_min, y_min, x_max, y_max,
                                 str_all_points_x, str_all_points_y, size, length, class_name]],
                               columns=['id', 'original_slide_name', 'x_min', 'y_min', 'x_max', 'y_max',
                                        'all_points_x', 'all_points_y', 'size', 'length', 'class'],
                               index=[str(index)])
        return new_row

    @staticmethod
    # pickleファイルのクラスIDからクラス名を取得
    def _get_class_names_from_id(id_array, class_list):
        index = id_array.tolist()
        class_names = []
        for i in index:
            class_names.append(class_list[i])
        return class_names

    @staticmethod
    def _get_all_points_xy_from_mask(mask):
        """
        maskから輪郭の座標情報を抽出 (load_from_pickle)
        :param mask: binary mask (検出結果のpickleファイルと同じ形式のマスク)
        :return all_points_x, all_points_y:
        """
        binary_mask = 255 * mask
        # binary_mask = 255 * mask[112:-122, 112:-112]  # 1024x1024で学習済みの場合
        binary_mask = binary_mask.astype(np.uint8)
        # binary_mask = cv2.resize(binary_mask, dsize=(640, 640))  # 1024x1024で学習済みの場合
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_points_x = []
        all_points_y = []
        if not len(contours) == 0:
            for point in contours[0]:
                all_points_x.append(point[0][0])
                all_points_y.append(point[0][1])
        all_points_x = np.array(all_points_x)
        all_points_y = np.array(all_points_y)
        return all_points_x, all_points_y

    @staticmethod
    def get_sub_df(df, conditions, col_name, include=True):
        """
        DataFrameから特定の条件のDataFrameを抽出
        :param df: DataFrame (pd.DataFrame)
        :param conditions: 条件
        None: col_nameが空のレコード
        ex) {'class': ['tooth', 'denticle']} : 'classの列がtoothまたはdenticleのレコード
        :param col_name: 対象の列名
        :param include: True:  特定の条件を満たすレコードのみ抽出
                        False: 特定の条件を満たさないレコードを抽出
        :return sub_df:　抽出したDataFrame
        """
        if conditions is None:
            sub_df = df[df[col_name].isnull()]
        elif len(conditions) == 0:
            sub_df = df
        else:
            sub_df = df
            for col, values in conditions.items():
                if include:
                    sub_df = sub_df[sub_df[col].isin(values)]
                else:
                    sub_df = sub_df[~sub_df[col].isin(values)]
        return sub_df

    @staticmethod
    def convert_to_vertex(all_points_x, all_points_y):
        """
        all_points_xとall_points_yから各頂点の座標の形式に変換する (put_labels)
        :param all_points_x:
        :param all_points_y:
        :return:
        """
        pts = []
        for x, y in zip(all_points_x, all_points_y):
            pts.append([x, y])
        pts = np.array([pts])
        return pts

    @staticmethod
    def display_operation(labels):
        """
        ラベル付けの際に表示する処理
        :param labels:
        :return:
        """
        string = ''
        for i, label in enumerate(labels):
            string += f'{label} ... \'{i}\', '
        sys.stdout.write(f'{string}\n')
        sys.stdout.write('ひとつ前に戻る ... \'b\', 次へ進む ... \'n\', 終了 ... \'q\'\n')
        print()
        clear_output(wait=True)

    @staticmethod
    def _get_mask(df, index, label_col, start_h, start_w):
        mask = dict()
        all_points_x = np.array(list(map(int, df.loc[index, 'all_points_x'].split(', '))))
        all_points_y = np.array(list(map(int, df.loc[index, 'all_points_y'].split(', '))))
        mask['all_points_x'] = (all_points_x - start_w).tolist()
        mask['all_points_y'] = (all_points_y - start_h).tolist()
        mask['name'] = df.loc[index, label_col]
        return mask

    @staticmethod
    def _make_shape_attributes(mask):
        shape_attributes = {"name": "polygon", "all_points_x": mask['all_points_x'],
                            "all_points_y": mask['all_points_y']}
        return shape_attributes

    @staticmethod
    def _make_region_attributes(mask):
        region_attributes = {"name": mask['name'], "colloquial_name": ""}
        return region_attributes


def excel2img(ROOT_DIR, excel_path, conditions_dict, save_dir, margin = 10, max_num = 9999):
    df = pd.read_excel(excel_path, index_col=0)
    for k, v in conditions_dict.items():
        assert type(v) in [int, float, str, list], 'condition_dict.values() should be either int, float, str, list'
        if type(v) in [int, float, str]:
            df = df[df[k] == v]
        elif type(v) == list:
            df = df[df[k].isin(v)]

    if len(df) > max_num:
        df = df[np.argsort(df['score']) < max_num]
    
    previous_filename = ''
    slide_image = np.array([0, 0])
    for index, item in df.iterrows():
        original_slide_name = item['original_slide_name']
        slide_image, previous_filename = load_slide_image(ROOT_DIR, original_slide_name, previous_filename, slide_image)
        x_min, y_min, x_max, y_max = tuple(item[['x_min', 'y_min', 'x_max', 'y_max']])
        cropped_img, savename = cropImg(slide_image, x_min, y_min, x_max, y_max, original_slide_name, margin = margin)
        imsave(os.path.join(save_dir, savename), cropped_img)

def load_slide_image(ROOT_DIR, filename, previous_filename, previous_slide_image):
    if filename != previous_filename:
        slide_path = glob(os.path.join(ROOT_DIR, 'data', 'images', '*', '*', filename))[0]
        slide_image = imread(slide_path)
        previous_filename = filename
    else:
        slide_image = previous_slide_image
    return(slide_image, previous_filename)

def cropImg(slide_image, x_min, y_min, x_max, y_max, filename, margin = 10):
    y_start = max(y_min - margin, 0)
    y_end = min(y_max + margin, slide_image.shape[0])
    x_start = max(x_min - margin, 0)
    x_end = min(x_max + margin, slide_image.shape[1])
    cropped_img = slide_image[y_start:y_end, x_start:x_end]
    savename = f'{filename[:-4]}_{y_start}_{x_start}.jpg'
    return(cropped_img, savename)