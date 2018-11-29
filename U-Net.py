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
from keras.layers import *
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.activations import relu
from keras import backend as K
from dataflow import DataLoader
import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
img_width = 256
img_height =256
img_channels = 3
batch_size = 8
learning_rate = 1e-4
epochs = 20000
img_path = 'trainB'
trans_path = 'transmission'
model_path = 'models'
def DehazeLoss(y_true, y_pred, alpha = 1, beta = 1):
    THRESHOLD = K.variable(1.0)
    mae = K.abs(y_true - y_pred)
    flag = K.greater(mae, THRESHOLD)
    l1 = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)))
    l2 = K.mean(K.square(y_pred-y_true))
    return alpha*l1+beta*l2

def DehazeLoss_std(y_true, y_pred, alpha = 1.0, beta = 1.0, batch_size = batch_size):
    std = 0
    total = 0
    THRESHOLD = K.variable(1.0)
    for i in range(batch_size):
        yt = y_true[i,:,:,:]
        yp = y_pred[i,:,:,:]
        mae = K.abs(yt - yp)
        flag = K.greater(mae, THRESHOLD)
        l1_temp = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)))
        l2_temp = K.mean(K.square(yt - yp))
        std_temp = K.std(yt)
        total += std_temp*(alpha*l1_temp+beta*l2_temp)
        std += std_temp
    return total/(std+K.epsilon())

class My_model():
    def __init__(self):
        self.gf = 16
        self.img_rows = img_height
        self.img_cols = img_width
        self.img_channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.img_path = img_path
        self.dataloader = DataLoader(self.img_path)
        optimizer = Adam(learning_rate, 0.5)
        self.model = self.build_model()
        print('Interupt...')
        self.model_path = model_path
        print(self.model.summary())
        self.model.compile(loss=DehazeLoss,
                           optimizer=optimizer)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def build_model(self):

        def conv2d(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same',activation='relu')(layer_input)
            d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input((None, None, self.img_channels))
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        # u4 = deconv2d(u3, d0, self.gf)
        u4 = UpSampling2D(size=2)(u3)

        output_img = Conv2D(self.img_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        return Model(d0, output_img)

    def train(self, epochs, batch_size=2, sample_interval=50):
        start_time = datetime.datetime.now()
        loss = 1000
        model_files = os.listdir(self.model_path)
        filename = ''
        minv = 100
        if len(model_files) >0:
            for file in model_files:
                l=file.split('.')[0].split('_')[-1]
                if int(l)<=minv:
                    minv=int(l)
                    filename=file
            self.model=load_model(os.path.join(self.model_path,'model_1200_0.0383.h5'), custom_objects={'DehazeLoss': DehazeLoss})
            # 自定义loss函数无法使用，在load_model函数里面增加参数custom_objects={'function_name':function}
            print('===========loading pretrained model===========')
        for epoch in range(epochs):
            for batch_i, (img, trans) in enumerate(self.dataloader.load_batch(batch_size)):
                loss = self.model.train_on_batch(img, trans)
                elapsed_time = datetime.datetime.now()-start_time
                print("[Epoch %d/%d] [Batch %d/%d] [Loss:%f] time: %s" %(epoch, epochs, batch_i, self.dataloader.n_batches, loss,elapsed_time))
                if batch_i % sample_interval ==0:
                    self.sample_images(epoch, batch_i)
            if epoch%100 == 0 :
                self.model.save(os.path.join(self.model_path,'model_%d_%.4f.h5'%(epoch, loss)))

    def sample_images(self, epoch, batch_i):
        os.makedirs('res', exist_ok=True)
        r, c = 3, 3
        img, syn = self.dataloader.load_data(batch_size=3, is_testing=True)
        res = self.model.predict(img)
        # res = np.array(res).reshape(None, 224, 224, 1)
        img = 0.5*img+0.5
        syn = 0.5*syn+0.5
        res = 0.5*res+0.5
        gen_imgs = [img, syn, res]

        titles = ['synthetise', 'image', 'restored']
        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                img = np.array(gen_imgs[i][j, :, :, :]).reshape(img_width, img_height, -1)
                img = np.minimum(np.maximum(img, 0.0), 1.0)
                if img.shape[-1] == 1:
                    img=np.reshape(img, (img_width, img_height))
                axs[i, j].imshow(img)
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
        fig.savefig('res\\%d_%d.jpg'%(epoch, batch_i))
        plt.close()

if __name__=='__main__':
    model = My_model()
    model.train(epochs=epochs, batch_size=batch_size, sample_interval=200)