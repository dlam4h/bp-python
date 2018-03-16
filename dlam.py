# -*- coding: UTF-8 -*-
import numpy as np
import datetime
import os
import pickle
np.random.seed(2018)

def Sigmoid(x):
    return 1/(1+np.exp(-x))


def Sigmoid_prime(x):
    return Sigmoid(x)*(1-Sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - np.tanh(x)*np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return 1. * (x > 0)


class Network(object):
    #初始化 网络结构 数据集 学习率 激活函数
    def __init__(self, net, data, activation='sigmoid'):
        self.net = net 
        self.data = data
        self.lr = 0.01
        self.loss = []
        self.datalen = len(data)
        self.biases = [np.random.randn(1, i) for i in net[1:]]
        self.weights = [np.random.randn(i,j) for i,j in zip(net[:-1],net[1:])]
        if activation == 'sigmoid':
            self.activation = Sigmoid
            self.activation_prime = Sigmoid_prime
        elif activation =='tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation =='relu':
            self.activation = relu
            self.activation_prime = relu_prime
        self.values = []
        self.active_values = []

    #前向传播
    def forward(self, a):
        self.values = []
        self.active_values = [a]
        i = 0
        for w,b in zip(self.weights,self.biases):
            a = np.dot(a,w) + b
            self.values.append(a)
            a = self.activation(a)
            self.active_values.append(a)
            i += 1
        return a

    #测试准确率时候的前向传播
    def test_forward(self, a):
        for w,b in zip(self.weights,self.biases):
            a = self.activation(np.dot(a,w) + b)
        return a

    #训练 次数 批处理大小 学习率 是否打乱数据
    def train(self,epochs,batch_size,lr = 0.01,shuffle = False):
        self.lr = lr
        for epoch in range(epochs):
            print('Epoch {}:'.format(epoch))
            a=[i for i in range(self.datalen)]
            if shuffle:
                np.random.shuffle(a)
            for i in range(0,self.datalen,batch_size):
                if i+batch_size<self.datalen:
                    batch_data, label = self.get_batch(a[i:i+batch_size])
                else:
                    batch_data, label = self.get_batch(a[i:self.datalen])
                self.forward(batch_data)
                self.update_batch(label,len(batch_data))
            print("Loss: {} || Acc: {}".format(self.loss[-1],self.evaluate(self.data)/self.datalen))

    #根据每个batch来进行梯度下降
    def update_batch(self,label,batch_size):
        new_weights = self.weights
        new_biases = self.biases
        self.loss.append(((self.active_values[-1]-label) ** 2).mean() / 2)
        delta = (self.active_values[-1]-label)*self.activation_prime(self.values[-1])
        new_biases[-1] = np.reshape((np.sum(delta, axis=0) / len(delta)), (1, self.net[-1]))
        # new_biases[-1]=delta
        new_weights[-1]=np.dot(self.active_values[-2].T,delta)
        for i in range(2,len(self.net)):
            delta = np.dot(delta,self.weights[-i+1].T)*self.activation_prime(self.values[-i])
            new_biases[-i]=np.reshape((np.sum(delta, axis=0) / len(delta)), (1, self.net[i-1]))
            # new_biases[-i] = delta
            new_weights[-i] = np.dot(self.active_values[-i-1].T,delta)
        for i in range(len(new_weights)-1):
            self.weights[i] = self.weights[i] - self.lr*new_weights[i]/batch_size
            self.biases[i] = self.biases[i] - self.lr*new_biases[i]/batch_size

    #获取每个batch数据
    def get_batch(self,numlist):
        data = []
        label = []
        for i in numlist:
            a = self.data[0][i]
            b = self.data[1][i]
            # a, b = self.data[i]
            data.append(a)
            label.append(b)
        data=np.asarray(data)
        label=np.asarray(label)
        return data,label

    #加载权重
    def load_weight(self, model):
        with open(model, 'rb') as f:
            (weight, biase) = pickle.load(f)
        self.weights = weight
        self.biases = biase

    #保存权重
    def save_model(self, save_path = ''):
        model = (self.weights,self.biases)
        timenow = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        name = timenow + '.pickle'
        output = open(os.path.join(save_path, name), 'wb')
        pickle.dump(model, output)
        output.close()

    #测试并输出准确率
    def evaluate(self, data, tr=True):
        a,b = data
        if tr:
            test_results = [(np.argmax(self.test_forward(x)), np.argmax(y)) for (x, y) in zip(a,b)]
        else:
            test_results = [(np.argmax(self.test_forward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in test_results)






