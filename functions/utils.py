import os
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


# for encording json files
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_pickle(obj, save_pickle_path):
    os.makedirs(os.path.dirname(save_pickle_path), exist_ok=True)
    with open(save_pickle_path, 'wb') as f:
        pickle.dump(obj, f, protocol=-1)


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


def concatenate_pickle(pickle_path_list):
    new_data = []
    for pickle_path in pickle_path_list:
        pickle_data = load_pickle(pickle_path)
        new_data.append(pickle_data[0])
    return new_data


def save_json(obj, save_json_path):
    with open(save_json_path, 'w') as f:
        json.dump(obj, f, cls = MyEncoder)


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def concatenate_json(json_path_list, save_json_path):
    new_data = {}
    for json_path in json_path_list:
        data = load_json(json_path)
        new_data.update(data)
    save_json(new_data, save_json_path)


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def crop_image(img_path, start_h, start_w, img_size, save_path):
    image = cv2.imread(img_path)
    cropped = image[start_h:start_h+img_size[0], start_w:start_w+img_size[1]]
    cv2.imwrite(save_path, cropped)

# show BGR (or grayscale) images in matplotlib figure
# scale of the image can be added in yaxis by setting parameter 'pixel2micron'
# (見邨追記)
def cv2Plt(cv2_image, figsize = (5, 5), pixel2micron = 2.54, show_scale = True):
    assert cv2_image is not None, 'image is NoneType'
    if len(cv2_image.shape) == 2:
        img = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    fx, fy = pixel2micron, pixel2micron
    img = cv2.resize(img, dsize = None, fx = fx, fy = fy)
    plt.figure(figsize = figsize)
    plt.imshow(img)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    for position in ['left', 'right', 'top', 'bottom']:
        ax.spines[position].set_visible(False)
    if not show_scale:
        ax.axes.yaxis.set_visible(False)
    plt.show()

def predict_contour(img):
    """
    find contours by threshoulding（見邨）
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 2)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # if multiple contours exists, return largest contour
    if len(contours) == 0:
        contour = contours 
    elif len(contours) == 1:
        contour = contours[0][:, 0, :]
    else:
        areas = [cv2.contourArea(ctr) for ctr in contours]
        contour = contours[np.argmax(areas)][:, 0, :]
    return(contour)

def load_xml(xml_path):
    # load xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_name_list = []
    x_min_list, y_min_list, x_max_list, y_max_list = [], [], [], []


    for Object in [child for child in root if child.tag == "object"]:
        # get class label
        class_name_list.append(Object[0].text)
        
        # get position in split image
        x_min_list.append(int(Object[4][0].text))
        y_min_list.append(int(Object[4][1].text))
        x_max_list.append(int(Object[4][2].text))
        y_max_list.append(int(Object[4][3].text))

    return(class_name_list, x_min_list, y_min_list, x_max_list, y_max_list)

def generate_json_template(json_data, img_path):
    img_name = os.path.basename(img_path)
    size = os.path.getsize(img_path)
    key = f'{img_name}{size}'
    subData = {'fileref': '', 'size': size, 'filename':img_name, 
               'base64_img_data':'', 'file_attributes':{}, 'regions':{}}
    json_data[key] = subData
    return(json_data)


# convert mask image to json format（見邨）
def mask2points(mask):
    assert len(mask.shape) == 2, 'function mask2points is not for color images'
    mask_255 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours((mask_255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) >= 1:
        all_points_x, all_points_y = contours[0][:, 0, :].T
    else:
        all_points_x, all_points_y = np.array([]), np.array([])
    return(all_points_x, all_points_y)

def masks2points_list(masks):
    num_masks = masks.shape[2]
    all_points_x_list, all_points_y_list = [], []
    for i in range(num_masks):
        mask = masks[:, :, i]
        all_points_x, all_points_y = mask2points(mask)
        all_points_x_list.append(all_points_x)
        all_points_y_list.append(all_points_y)
    return(all_points_x_list, all_points_y_list)

def SquareResize(img, size, background = 'mean'):
    # 入力された画像を，指定したサイズの正方形にリサイズする
    # 出力はカラー画像とする
    # 縦横比はそのまま残し，余白はグレーで埋める
    h, w = img.shape[:2]
    mag = size / max(h, w)
    img2 = cv2.resize(img ,(int(w * mag), int(h * mag)))
    
    if background == 'mean':
            background = int(np.mean(img))
    img3 = np.ones((size, size, 3), dtype = "u1") * background
    if h > w:
        img3[0 : img2.shape[0], int((size - img2.shape[1]) / 2): int((size + img2.shape[1]) / 2)] = img2
    else:
        img3[int((size - img2.shape[0]) / 2) : int((size + img2.shape[0]) / 2), 0:img2.shape[1]] = img2
    
    return(img3)