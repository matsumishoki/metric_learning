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

def softs(num_train_small_data,rank_labels,T_data, softs_K):
    soft_accuracy = []
    softs_accuracies = []
    for soft_k in softs_K:
        rank_soft_k_rabels = rank_labels[:,:soft_k]
        cheak_True_or_False = []
        for i in range(num_train_small_data):
            soft_top = T_data[i]==rank_soft_k_rabels[i]
            cheak_True_or_False.append(np.any(soft_top))
        average_soft_top_accuracy = (np.count_nonzero(cheak_True_or_False)/num_train_small_data)*100
        soft_accuracy.append(average_soft_top_accuracy)
    softs_accuracies.append(soft_accuracy)
    return softs_accuracies

def making_top_k_data(D,labels,K):
        sorted_D=[]
        rank_labels=[]
        for d_i in D:
            top_k_indexes = np.argpartition(d_i, K)[:K]
            sorted_top_k_indexes = top_k_indexes[np.argsort(d_i[top_k_indexes])]
            sorted_D.append(sorted_top_k_indexes)
            ranked_label = labels[sorted_top_k_indexes]
            # 最初の距離は0であるため除去する
            rank_labels.append(ranked_label[1:])
        return np.array(rank_labels)


def hards(num_train_small_data,rank_labels,T_data):
    cheak_True_or_False = []
    for i in range(num_train_small_data):
        hard_top = T_data[i]==rank_labels[i]
        cheak_True_or_False.append(np.all(hard_top))
    average_hard_top_accuracy = (np.count_nonzero(cheak_True_or_False)/num_train_small_data)*100 
    return average_hard_top_accuracy


def retrievals(num_train_small_data,rank_labels,T_data):
    cheak_True_or_False = []
    for i in range(num_train_small_data):
        retrievals_top = T_data[i]==rank_labels[i]
        cheak_True_or_False.append(np.mean(retrievals_top))
    average_retrievals_top_accuracy = (np.count_nonzero(cheak_True_or_False)/num_train_small_data)*100 
    return average_retrievals_top_accuracy
