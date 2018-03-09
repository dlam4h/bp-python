# -*- coding: UTF-8 -*-
from dlam import Network
import numpy as np
if __name__ == '__main__':
    #数据格式 M*N M表示样本数 N表示数据
    #标签转化为只有0和1的数字
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    Y = np.array([[1,0],[0,1],[1,0],[0,1]])
    data=[]
    data.append(X)
    data.append(Y)
    dlam = Network([3,4,4,2], data)
    dlam.train(5, 4, 0.1, shuffle=False)