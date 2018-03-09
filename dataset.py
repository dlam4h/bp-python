# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import os
import pandas as pd
def load_data(data_root, net_shape,resize=None, test_data=None):
    data=[]
    file = os.listdir(data_root)
    for i in range(len(file)):
        imgs=os.listdir(os.path.join(data_root,file[i]))
        for img in imgs:
            image = cv2.imread(os.path.join(data_root,file[i],img))
            if resize:
                image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
            img_data = np.reshape(image, (net_shape[0], 1))
            if test_data:
                a=i
            else:
                a = np.array([np.array([0],dtype=np.int64) for j in range(net_shape[-1])])
                a[i][0] = 1
            data.append((img_data,a))
    return data
#从csv读取
class Load_data_csv(object):
    def __init__(self, data_name,net_shape,test_data=None):
        self.net_shape = net_shape
        self.test_data  = test_data
        train = pd.read_csv(data_name)
        self.X = train.values[:, 1:]
        self.label = train.values[:, 0]

    def __getitem__(self, item):
        dlam=[]
        x = self.X[item]
        y = self.label[item]
        x = np.reshape(x, (len(x), 1))
        if self.test_data:
            a = y
        else:
            a = np.array([np.array([0], dtype=np.int64) for k in range(self.net_shape[-1])])
            a[y][0] = 1
        dlam.append((x,a))
        return dlam

    def __len__(self):
        return len(self.X)

def load_data_csv(data_name,net_shape,test_data=None):
    dlam = []
    train = pd.read_csv(data_name)
    X = train.values[:, 1:]
    label = train.values[:, 0]
    for x, y in zip(X, label):
        x = np.reshape(x, (len(x), 1))
        if test_data:
            a = y
        else:
            a = np.array([np.array([0], dtype=np.int64) for j in range(net_shape[-1])])
            a[y][0] = 1
        dlam.append((x, a))
    return dlam
