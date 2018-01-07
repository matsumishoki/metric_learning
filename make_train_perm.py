# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:59:00 2018

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
import MNIST_convnet as M
from sklearn.metrics import pairwise_distances


def make_train_perm_data(T_train):
    T_train = np.sort(T_train)
    num_train = len(T_train)
    i_0 = []
    i_1 = []
    i_2 = []
    i_3 = []
    i_4 = []
    i_5 = []
    i_6 = []
    i_7 = []
    i_8 = []
    i_9 = []
    for i in range(num_train):
        t_data = T_train[i]
        if t_data == 0:
            i_0.append(i)
            num_0 = len(i_0) - 1
        if t_data == 1:
            i_1.append(i)
            num_1 = num_0 + len(i_1)
        if t_data == 2:
            i_2.append(i)
            num_2 = num_1 + len(i_2)
        if t_data == 3:
            i_3.append(i)
            num_3 = num_2 + len(i_3)
        if t_data == 4:
            i_4.append(i)
            num_4 = num_3 + len(i_4)
        if t_data == 5:
            i_5.append(i)
            num_5 = num_4 + len(i_5)
        if t_data == 6:
            i_6.append(i)
            num_6 = num_5 + len(i_6)
        if t_data == 7:
            i_7.append(i)
            num_7 = num_6 + len(i_7)
        if t_data == 8:
            i_8.append(i)
            num_8 = num_7 + len(i_8)
        if t_data == 9:
            i_9.append(i)
#            num_9 = num_8 + len(i_9)

    m_0 = np.random.permutation(i_0)[:100]
    m_1 = np.random.permutation(i_1)[:100]
    m_2 = np.random.permutation(i_2)[:100]
    m_3 = np.random.permutation(i_3)[:100]
    m_4 = np.random.permutation(i_4)[:100]
    m_5 = np.random.permutation(i_5)[:100]
    m_6 = np.random.permutation(i_6)[:100]
    m_7 = np.random.permutation(i_7)[:100]
    m_8 = np.random.permutation(i_8)[:100]
    m_9 = np.random.permutation(i_9)[:100]
    train_data = list(np.vstack((m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9)))
    return train_data
    
#if __name__ == '__main__':
#    X_train, T_train, X_test, T_test = load_mnist.load_mnist()
#    T_train = T_train.astype(np.int32)
#    T_test = T_test.astype(np.int32)
#
#
#    
#     # 60000ある訓練データセットを54000と6000の評価のデータセットに分割する
#    X_train, X_valid, T_train, T_valid = train_test_split(
#        X_train, T_train, test_size=0.1, random_state=100)

    ##ここから関数にする

#    print("num_0:", num_0)
#    print("num_1:", num_1)
#    print("num_2:", num_2)
#    print("num_3:", num_3)
#    print("num_4:", num_4)
#    print("num_5:", num_5)
#    print("num_6:", num_6)
#    print("num_7:", num_7)
#    print("num_8:", num_8)
#    print("num_9:", num_9)