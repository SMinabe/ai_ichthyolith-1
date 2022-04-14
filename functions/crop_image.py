import os

from functions.utils import crop_image


img_path = '../data/images/MR15_E01_PC11/24_05_26/MR15_E01_PC11_24_05_26_002.jpg'
start_h = 3456
start_w = 7488
save_dir = '../data/images/annotations/'
save_img_name = os.path.basename(img_path).replace('.jpg', f'_{str(start_h).zfill(2)}_{str(start_w).zfill(2)}.jpg')
save_path = os.path.join(save_dir, save_img_name)
img_size = (640, 640)

save_path = '../data/images/presentation3.jpg'
crop_image(img_path, start_h, start_w, img_size, save_path)
