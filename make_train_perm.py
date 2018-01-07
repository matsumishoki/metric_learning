# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:59:00 2018

@author: matsumi
"""

import numpy as np


def make_train_perm_data(T_train, data_size):
    num_examples = len(T_train)
    num_classes = len(np.unique(T_train))
    
    indexes = []
    for c in range(num_classes):
        indexes_c = np.arange(num_examples)[T_train == c]
        random_indexes_c = np.random.choice(indexes_c, data_size, False)
        indexes.append(random_indexes_c)
    return np.concatenate(indexes)
