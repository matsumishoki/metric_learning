# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:31:49 2018

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import copy
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.optimizers
from chainer import cuda
from chainer import Variable, Chain, optimizers
from chainer.cuda import cupy


class ConvNet(Chain):
        def __init__(self):
            super(ConvNet, self).__init__(
                    conv_1 = L.Convolution2D(1, 50, 5),
                    conv_12 = L.Convolution2D(50, 50, 1),
                    conv_2 = L.Convolution2D(50, 100, 5),
                    conv_3 = L.Convolution2D(100, 200, 4),
                    bn_1 = L.BatchNormalization(50),
                    l_1 = L.Linear(200, 400),
                    l_2 = L.Linear(400, 10),
                    )
        def __call__(self, x_data, train):
            x = Variable(x_data.reshape(-1, 1, 28, 28))
            h = self.conv_1(x)
            h = self.conv_12(h)
            h = F.max_pooling_2d(h, 2)
            h = F.relu(h)
            h = self.conv_2(h)
            h = F.max_pooling_2d(h, 2)
            h = F.relu(h)
            h = self.conv_3(h)
            h = F.relu(h)
            h = self.l_1(h)
            h = F.relu(h)
            Y = self.l_2(h)
            return Y



def metric_loss_average(model, x_data, t_data, num_batches, train):
#    accuracies = []
    losses = []
    total_data = np.arange(len(x_data))
    
    for indexes in np.array_split(total_data, num_batches):
        X_batch = cuda.to_gpu(x_data[indexes])
        T_batch = cuda.to_gpu(t_data[indexes])
#        print("a")        
        with chainer.using_config('train', train):
            # 順伝播させる
            Y = model.__call__(X_batch, train)
        Y1, Y2 = F.split_axis(Y,2, axis=0)
        T1, T2 = F.split_axis(T_batch,2, axis=0)
        T = (T1.array==T2.array).astype(np.int32) 
        loss = F.contrastive(Y1, Y2, T)
        loss_cpu = cuda.to_cpu(loss.data)
        losses.append(loss_cpu)
    return np.mean(losses)
