#-*- coding:utf-8 -*-

# author:小霸王游戏机

# datetime:2018/9/26 13:06

import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import keras.backend as K
class DataLoader():

    def __init__(self, img_path, img_res=(256, 256)):
        self.img_path = img_path
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        path = glob(os.path.join(self.img_path, '*.jpg'))
        img_H, img_W = self.img_res

        random.shuffle(path)
        batch_images = np.random.choice(path, size=batch_size)

        img = []
        hazy_img = []
        for img_path in batch_images:
            image = self.imread(img_path)

            h, w, _ = np.array(image).shape

            H = np.random.random()*(h-img_H)
            W = np.random.random()*(w-img_W)

            box = (W, H, W+img_W, H+img_H)

            image = np.array(image.crop(box), dtype=float)
            image = np.reshape(image, (img_H, img_W, 3))
            t = 0.3 + 0.6*np.random.random()
            A = 255.0 * 0.7 + np.random.random() * 255.0 * 0.3
            haze_img = image * t + A * (1 - t)
            # print(haze_img)
            # If training => do random rotate
            # data augmentation

            if not is_testing:
                # if np.random.random() < 0.5:
                #     img = np.fliplr(img)
                #     trans = np.fliplr(trans)
                if np.random.random() < 0.5:
                    image = np.rot90(image)
                    haze_img = np.rot90(haze_img)
                if np.random.random() < 0.5:
                    image = np.rot90(image)
                    haze_img = np.rot90(haze_img)
                if np.random.random() < 0.5:
                    image = np.rot90(image)
                    haze_img = np.rot90(haze_img)
                if np.random.random() < 0.5:
                    image = np.rot90(image)
                    haze_img = np.rot90(haze_img)

            img.append(image)
            hazy_img.append(haze_img)

        img = np.array(img, dtype=float)/127.5 - 1.0
        hazy_img = np.array(hazy_img, dtype=float)/127.5 - 1.0


        return hazy_img, img

    def load_batch(self, batch_size=1, is_testing=False):

        path = glob(os.path.join(self.img_path, '*.jpg'))
        img_H, img_W = self.img_res
        random.shuffle(path)
        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            img, hazy_img = [], []
            for img_path in batch:
                image = self.imread(img_path)

                h, w, _ = np.array(image).shape

                H = int(np.random.random() * (h - img_H))
                W = int(np.random.random() * (w - img_W))

                # box = (H, W, H + img_H, W + img_W)
                box = (W, H, W + img_W, H + img_H)
                image = np.array(image.crop(box),dtype=float)
                image = np.reshape(image, (img_H, img_W, 3))
                t = np.random.random()
                A = 255.0 * 0.7 + np.random.random() * 255.0 * 0.3

                haze_img = image * t + A * (1.0 - t)
                # If training => do random flip
                if not is_testing:
                    # if np.random.random() < 0.5:
                    #     img = np.fliplr(img)
                    #     trans = np.fliplr(trans)
                    if np.random.random() < 0.5:
                        image = np.rot90(image)
                        haze_img = np.rot90(haze_img)
                    if np.random.random() < 0.5:
                        image = np.rot90(image)
                        haze_img = np.rot90(haze_img)
                    if np.random.random() < 0.5:
                        image = np.rot90(image)
                        haze_img = np.rot90(haze_img)
                    if np.random.random() < 0.5:
                        image = np.rot90(image)
                        haze_img = np.rot90(haze_img)

                img.append(image)
                hazy_img.append(haze_img)

            img = np.array(img, dtype=float) / 127.5 - 1.0
            hazy_img = np.array(hazy_img, dtype=float) / 127.5 - 0.5

            yield hazy_img, img
    def imread(self, img_path):
        return Image.open(img_path)


if __name__=='__main__':
    dataloader = DataLoader('trainB')

    img, hazy_img = dataloader.load_data(batch_size=3)

    x = K.variable(img)
    y = K.variable(hazy_img)
    THRESHOLD = 1.0
    mae = K.abs(y - x)
    flag = K.greater(mae, THRESHOLD)
    l1 = K.switch(flag, (mae - 0.5), K.pow(mae, 2))
    l2 = K.square(y - x)
    sess = K.get_session()

    print(sess.run(K.std(x[0,:,:,:])))
    a = np.array(sess.run(x[0,:,:,:]))
    a = np.reshape(a, (224,224,3))

    print(a.shape)
    fig = plt.figure()
    plt.imshow((a+1.0)/2.0)
    plt.show()
    # print(sess.run(l2))
    # print(np.mean(np.square(img[:,:,:,0] - np.mean(img[:,:,:,0]))))

