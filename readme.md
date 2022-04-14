## Citation
Mimura, K., Minabe, S., Nakamura, K., Yasukawa, K., Ohta, J., Kato, Y. Automated detection of microfossil fish teeth from slide images using combined deep-learning models. *submitted to Computers & Geosciences*

# About
**ai_ichthyolith** is an application of Mask R-CNN for detecting microfossil (typically microfossil fish teeth called **ichthyolith**) images.

<img src="/images_for_github/detect_example.jpg" width="300">
<br>

## What is new?

**Object detection** is useful for observing and counting various kinds of signals from a broad area by taking a number of pictures.
However, **objects located at image boundaries** are not captured in full shape, which can hamper observation. 
This problem can be solved by setting an overlap in images, but this in turn raise another problem of **duplicated detections**.

![image2](/images_for_github/overlap.png)

<br>

This program aims to detect objects in its complete form without duplications by converting their x and y coordinates of the object in each image (**relative coordinates**) into the coordinates in the entire imaging area (**absolute coordinates**).

<details><summary> <span style="font-size: 130%">
日本語
</span></summary><div>

多数の画像を撮影して**物体検出**にかけることにより，広範囲から対象を観察したりカウントしたりすることが可能です．
しかし，範囲を分割して撮影すると，**画像の境界に存在する粒子**が完全な形状で撮影されないという問題があります．
この問題は，範囲に重なりを設定して撮影することで解決できますが，今度は１つの物体が**重複**して検出されるという別の問題が生じます
<br>

本プログラムでは，各画像中での x, y 座標（**相対座標**）を，全体の撮影範囲の中での X, Y 座標（**絶対座標**）に変換することにより，重複を防ぎながら完全な形状で検出することを目的としています．

</div></details>

<br>

# How to use?
## training and validation
A PC with GPU is required for training. We used online server [paperspace](https://www.paperspace.com/). We could not use [Google Colaboratory](https://colab.research.google.com/?hl=en) because training took more than 12 hours.

1. generate training and validation dataset following to the descriptions in [Mask_RCNN](https://github.com/matterport/Mask_RCNN/).
2. customize training settings by editing `/ichthyolith/ichthyolith_const.py` and `/ichthyolith/ichthyolith_setting.py`
3. move to directory "ichthyolith" and type `python ichthyolith_train.py --weights=coco && python ichthyolith_val.py` to command/

## detection
We used Google Colaboratory for detection.

1. Take images and name them by (sample name)\_(slide number)\_(absolute Y of top)\_(absolute_X of left).jpg (We highly reccomennd to name images automatically)
2. Store images at `/data/images/(site name)/(sample name)/(slide name)/~.jpg`
3. Open `/notebooks/detect_images.ipynb` and run all the cells.

## test
We used Google Colaboratory for test.

1. Store images for test. Format of image is same as that of detection images.
2. Do box annotation by  
3. Open `/notebooks/detect_test.ipynb` and run all the cells.


## Dataset availability
Training, validation and practical-test (see paper for detail) datasets are temporally available at Author's [Google Drive](https://drive.google.com/drive/folders/1QC7deWgQRFkoDdOLao3SatYC6TO4moeT?usp=sharing).

## for second-stage classification
To reduce false positives, we re-classify the detected regions by image classification model EfficientNet-V2. We provide sample codes that are customized for output format of ai_ichthyolith at [eNetV2_for_ai_ichthyolith](https://github.com/KazuhideMimura/eNetV2_for_ai_ichthyolith).

# References
He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask r-cnn. *In* Proceedings of the IEEE international conference on computer vision (pp. 2961-2969). [[GitHub](https://github.com/matterport/Mask_RCNN)] [[paper](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)]

# notes
## log
2022/4/14: released

## Todo
1. prepare eNet_for_ai_ichthyolith
3. submit paper to *Computers & Geosciences*
4. share preprint on EarthArXiv
5. add some images for example
6. translate Japanese comments to English. 
