Python 3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:54:40) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import cv2
Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'
>>> import numpy as np
>>> import os
>>> from random import shuffle
>>> from tqdm import tqdm
>>> TRAIN_DIR='C:\Users\Sohan Nipunage\Google Drive\Spring 2018\ML\Term Project\download_data\train_images'
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
>>> TRAIN_DIR="C:\Users\Sohan Nipunage\Google Drive\Spring 2018\ML\Term Project\download_data\train_images"
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
>>> TRAIN_DIR="C:/Users/Sohan Nipunage/Google Drive/Spring 2018/ML/Term Project/download_data/train_images"
>>> img_size=50
>>> IMG_SIZE=50
>>> LR=1e-3
>>> MODEL_NAME='landmarkRecog-{}-{}.model'.format(LR,'2conv-basic')
>>> def label_img(img):
	word_label=
