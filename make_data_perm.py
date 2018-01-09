# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:59:00 2018

@author: matsumi
"""

import numpy as np
import chainer.optimizers
from chainer import cuda
from sklearn.metrics import pairwise_distances


def make_data_perm_data(T_train, data_size):
    num_examples = len(T_train)
    num_classes = len(np.unique(T_train))
    
    indexes = []
    for c in range(num_classes):
        indexes_c = np.arange(num_examples)[T_train == c]
        random_indexes_c = np.random.choice(indexes_c, data_size, False)
        indexes.append(random_indexes_c)
    return np.concatenate(indexes)

def distance_and_T_data(X_data,T_data,extract_data, model):
    Y_data = []
    X_small_data = X_data[extract_data]
    T_small_data = T_data[extract_data]
    with chainer.no_backprop_mode():
        X_small_data = cuda.to_gpu(X_small_data)            
        y_train = model(X_small_data, False)
        Y_data.append(y_train.array)
    
    # Yから距離行列Dに変換する
    Y_data = cuda.to_cpu(Y_data)
    Y_data = np.vstack(Y_data)
    D = pairwise_distances(Y_data)
    return D, T_small_data


if __name__ == '__main__':
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    for i in range(10):
        indexes = make_data_perm_data(labels, 3)
        np.testing.assert_array_equal(labels[indexes],
                                      [0, 0, 0, 1, 1, 1, 2, 2, 2])