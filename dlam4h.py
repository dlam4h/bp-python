# -*- coding: UTF-8 -*-
from dlam import Network
import numpy as np
import pickle
import gzip
def get_label(x):
    a = [0 for i in range(10)]
    a[x] = 1.0
    return a
if __name__ == '__main__':
    #数据格式 M*N M表示样本数 N表示数据
    #标签转化为只有0和1的数字
    # example 1
    file = gzip.open('mnist.pkl.gz', 'rb')
    train_data, val_data, test_data = pickle.load(file, encoding='iso-8859-1')
    file.close()
    train = [np.reshape(x, (784)) for x in train_data[0]]
    label = [get_label(y) for y in train_data[1]]
    data = []
    data.append(np.array(train))
    data.append(np.array(label))
    dlam = Network([784, 100, 10], data)


    # example 2
    # X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    # Y = np.array([[1,0],[0,1],[1,0],[0,1]])
    # data=[]
    # data.append(X)
    # data.append(Y)
    # dlam = Network([3,4,2], data)


    dlam.train(300, 1000, 0.1, shuffle=True)