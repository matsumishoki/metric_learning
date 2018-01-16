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
from chainer.functions import matmul
from chainer.functions import transpose
from chainer.functions import softmax_cross_entropy
from chainer.functions import batch_l2_norm_squared


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


def make_n_pair(N, T_data):
    # 抽出するNクラスを選択する
    T_data_num_classes = np.unique(T_data)
    num_T_data = len(T_data)
    select_N_classes = np.random.choice(T_data_num_classes, N,replace=False)
    # 抽出するNクラスを選ぶ
    select_indexes = []
    for i in range(len(select_N_classes)):
        for j in range(num_T_data):
            if select_N_classes[i]==T_data[j]:
                select_indexes.append(j)
    # 添字データを扱いやすいようにする
    select_indexes = np.array((np.split(np.array(select_indexes),N)))
    pairs = []
    for i in range(N):
        pair = np.random.choice(select_indexes[i],2,replace=False)
        pairs.append(pair)
    pairs = np.concatenate(np.array(pairs))
    ank = pairs[0::2]
    positive = pairs[1::2]
    pairs = np.concatenate((ank, positive))
    return pairs


def n_pair_mc_loss(f, f_p, l2_reg):
    logit = matmul(f, transpose(f_p))
    N = len(logit.data)
    xp = cuda.get_array_module(logit.data)
    loss_sce = softmax_cross_entropy(logit, xp.arange(N))
    l2_loss = sum(batch_l2_norm_squared(f) + batch_l2_norm_squared(f_p))
    loss = loss_sce + l2_reg * l2_loss
    return loss


def n_pair_metric_loss_average(model, x_data, t_data, num_batches, train, loss_l2_reg):
#    accuracies = []
    losses = []
    total_data = np.arange(len(x_data))
    
    for indexes in np.array_split(total_data, num_batches):
        X_batch = cuda.to_gpu(x_data[indexes])
        T_batch = cuda.to_gpu(t_data[indexes])     
        with chainer.using_config('train', train):
            # 順伝播させる
            Y = model.__call__(X_batch, train)
        Y1, Y2 = F.split_axis(Y,2, axis=0)
        T1, T2 = F.split_axis(T_batch,2, axis=0)
        loss = n_pair_mc_loss(Y1, Y2, loss_l2_reg)
        loss_cpu = cuda.to_cpu(loss.data)
        losses.append(loss_cpu)
    return np.mean(losses)