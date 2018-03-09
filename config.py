# -*- coding: UTF-8 -*-
import warnings

class DefaultConfig(object):
    train_data_root = 'F:/Desktop/index'  # 训练集存放路径
    test_data_root = 'F:/Desktop/index'  # 测试集存放路径
    train_data_name = 'index.csv'
    test_data_name = 'index.csv'
    resize_shape = (28,28)
    net_shape = [784, 100, 10]
    epochs = 30
    batch_size = 20
    activation = 'sigmoid'
    lr = 0.1

def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))
DefaultConfig.parse = parse
opt = DefaultConfig()

#调用
#from config import DefaultConfig
# opt = DefaultConfig()
# opt.param
#修改
# new_config = {'net_shape':[3,4,4,2]}
# opt.parse(new_config)