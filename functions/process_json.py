import os
import json
from glob import glob

from functions.utils import save_json

def extract_sub_class_json(json_path, class_names, save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    sub_annotations = {}
    for key, value in data.items():
        i = 0
        regions = value['regions']
        sub_regions = {}
        for region in regions.values():
            if 'name' in region['region_attributes'].keys(): # 見
                name = region['region_attributes']['name']
                # print(key) # 見邨コメントアウト
                if name in class_names:
                    sub_regions[str(i)] = region
                    i += 1
        if len(sub_regions) != 0:
            value['regions'] = sub_regions
            sub_annotations[key] = value
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(sub_annotations, f)
    print(f'saved: {save_path}')


def convert_class_label(json_path, class_names_dict, save_path):
    
    # 見邨追加
    extract_sub_class_json(json_path, class_names_dict.keys(), save_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    new_data = dict()
    for key, value in data.items():
        regions = value['regions']
        for i in range(len(regions)):
            region_attributes = regions[str(i)]['region_attributes']
            if region_attributes == {}:
                value['regions'][str(i)]['region_attributes']['name'] = class_names_dict['']
            else:
                name = region_attributes['name']
                value['regions'][str(i)]['region_attributes']['name'] = class_names_dict[name]
        new_data[key] = value
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(new_data, f)
    print(f'saved: {save_path}')


def format_to_json(json_data, img_path, contours, class_name):
    """
    overwrite new data in json format（見邨）
    if save_json_path is valid, save json file
    """
    img_name = os.path.basename(img_path)
    size = os.path.getsize(img_path)
    key = img_name + str(size)
    
    regions = {}
    for k, ctr in enumerate(contours):
        all_points_x, all_points_y = ctr.T
        shape_attributes = {'name': 'polygon',
                            'all_points_x': list(all_points_x)[::3] + [all_points_x[0]],
                            'all_points_y': list(all_points_y)[::3] + [all_points_y[0]]}
        region_attributes = {'name': class_name}
        regions[str(k)] = {'shape_attributes': shape_attributes, 
                           'region_attributes': region_attributes}
        
    if len(regions) >= 1:
        json_data[key] = {'fileref': '',
                        'size': size,
                        'filename': img_name,
                        'base64_img_data': '',
                        'file_attributes': {},
                        'regions':regions}

    return(json_data)


if __name__ == '__main__':
    j_path = '../data/dataset/dataset_with_noise_320/train/tooth_noise.json'
    class_names = ['tooth']
    class_names_dict = {'tooth': 'tooth', 'denticle': 'noise', '': 'noise'}
    save_path = '../data/dataset/dataset_with_noise_320/train/tooth_only.json'
    extract_sub_class_json(j_path, class_names, save_path)
