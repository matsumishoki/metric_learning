# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:15:01 2017

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

if __name__ == '__main__':
    X_train, T_train, X_test, T_test = load_mnist.load_mnist()
    T_train = T_train.astype(np.int32)
    T_test = T_test.astype(np.int32)
    plt.matshow(X_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

    print ("X_train.shape:", X_train.shape)
    print ("T_train.shape:", T_train.shape)
    
     # 60000ある訓練データセットを54000と6000の評価のデータセットに分割する
    X_train, X_valid, T_train, T_valid = train_test_split(
        X_train, T_train, test_size=0.1, random_state=100)
    print ("X_train.shape:", X_train.shape)
    print ("T_train.shape:", T_train.shape)
    print ("X_valid.shape:", X_valid.shape)
    print ("T_valid.shape:", T_valid.shape)
    print ("X_test.shape:", X_test.shape)
    print ("T_test.shape:", T_test.shape)