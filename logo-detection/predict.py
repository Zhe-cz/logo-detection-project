from nets.yolo3 import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import cv2
import numpy as np
import os




yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image = np.asarray(r_image,dtype=np.uint8)
        target_path = './target_img'
        erode_demo(r_image, target_path)
yolo.close_session()
