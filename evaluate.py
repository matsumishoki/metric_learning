# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 02:29:48 2018

@author: matsumi
"""
import make_data_perm as mdp
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

def softs(num_train_small_data,rank_labels,T_data):
    cheak_True_or_False = []
    for i in range(num_train_small_data):
        soft_top = T_data[i]==rank_labels[i]
        cheak_True_or_False.append(np.any(soft_top))
    average_soft_top_accuracy = (np.count_nonzero(cheak_True_or_False)/num_train_small_data)*100 
    return average_soft_top_accuracy