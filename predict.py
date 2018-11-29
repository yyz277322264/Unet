#-*- coding:utf-8 -*-

# author:小霸王游戏机

# datetime:2018/9/26 13:05

import cv2
import numpy as np
import h5py
import os
from keras.applications.resnet50 import ResNet50
from keras.utils import np_utils,conv_utils
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
from dataflow import DataLoader
import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
from glob import glob
from PIL import Image
from model import *
img_width = 224
img_height =224
img_channels = 3
batch_size = 4
learning_rate = 1e-4
epochs = 200
img_path = 'image'
trans_path = 'transmission'
model_path = 'models'
result_path = 'result'
class Predict_model():
    def __init__(self):
        self.img_path = img_path
        self.model_path = model_path
        self.model = load_model(os.path.join(self.model_path, 'model_300_0.0795.h5'), custom_objects={'DehazeLoss': DehazeLoss})
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        print(self.model.summary())

    def imread(self, img_path):
        return Image.open(img_path)

    def load_data(self, img_path):

        haze_img = []
        img = self.imread(img_path)
        img = np.array(img)
        h, w, c = img.shape
        ori_h, ori_w = h, w

        for i in range(h, h + 32):
            if i % 32 == 0:
                h = i
                break
        for j in range(w, w + 32):
            if j % 32 == 0:
                w = j
                break
        new_img = np.zeros((h, w, c))
        new_img[:ori_h, :ori_w, :c] = img
        img = new_img
        haze_img.append(img)
        haze_img = np.array(haze_img) / 225.0 - 0.5
        return haze_img, ori_h, ori_w, img_path

    def predict(self, img, ori_h, ori_w, img_path):
        _,h,w,c=img.shape
        start_time = datetime.datetime.now()
        res=self.model.predict(img)
        end_time = datetime.datetime.now()
        elapsed_time = end_time-start_time
        print('%s s for one image'%elapsed_time)
        res = res + 0.5
        res = np.array(res)

        res = np.minimum(np.maximum(res, 0.0), 1.0)
        res = np.reshape(res, (h,w,-1))

        res*=255.0
        res = np.uint8(res)


        print(res.shape)
        # print(np.array(res).shape)
        res = res[:ori_h,:ori_w,:]
        im = Image.fromarray(res)
        if im.mode !='RGB':
            im = im.convert('RGB')
        save_path = os.path.join(self.result_path,img_path.split('\\')[-1])

        im.save(save_path)




if __name__=='__main__':
    model = Predict_model()
    files = glob(os.path.join(r'E:\test_images','*.jpg'))
    for file in files:
        img, ori_h, ori_w, img_path = model.load_data(file)
        model.predict(img,ori_h,ori_w,img_path)