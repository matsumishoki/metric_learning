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
import MNIST_convnet as M

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

    num_train = len(X_train)
    num_valid = len(X_valid)
    num_test = len(X_test)

    classes = np.unique(T_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = X_train.shape[-1]  # xの次元

    # 超パラメータの定義
    learning_rate = 0.0001  # learning_rate(学習率)を定義する
    max_iteration = 1      # 学習させる回数
    batch_size = 200       # ミニバッチ1つあたりのサンプル数
 
    model = M.ConvNet().to_gpu()

    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    loss_train_history = []
    train_accuracy_history = []
    loss_valid_history = []
    valid_accuracy_history = []

    valid_accuracy_best = 0
    valid_loss_best = 10
    num_batches = num_train / batch_size  # ミニバッチの個数
    num_valid_batches = num_valid / batch_size
    
    # 学習させるループ
    for epoch in range(max_iteration):
        print ("epoch:", epoch)
        w_1_grad_norms = []
        w_2_grad_norms = []
        w_3_grad_norms = []
        b_1_grad_norms = []
        b_2_grad_norms = []
        b_3_grad_norms = []
        #ループの時間を計測する
        time_start = time.time()
        # 入力画像の一枚目と二枚目を無作為に選ぶ
        perm_1 = np.random.permutation(num_train)
        perm_2 = np.random.permutation(num_train)
#        print("perm_1",perm_1)
#        print("perm_2",perm_2)
        # mini batchi SGDで重みを更新させるループ
        for batch_indexes in np.array_split(perm_1, num_batches):
            x_batch_1 = cuda.to_gpu(X_train[batch_indexes])
            t_batch_1 = cuda.to_gpu(T_train[batch_indexes])
        
        for batch_indexes in np.array_split(perm_2, num_batches):
            x_batch_2 = cuda.to_gpu(X_train[batch_indexes])
            t_batch_2 = cuda.to_gpu(T_train[batch_indexes])
                        
            # 勾配を初期化する            
            model.zerograds()
            # contrastive loss関数に入力するy_1,y_2を取得する
            y_1 = model.loss_and_accuracy(x_batch_1, t_batch_1, True)
            y_2 = model.loss_and_accuracy(x_batch_2, t_batch_2, True)
            # tが同じならば1，異なるならば0を入力する
            t=[]
            for element in range(batch_size):
                if t_batch_1[element] == t_batch_2[element]:
                    t.append(1)
                else:
                    t.append(0)
            print("t",t)
#            print("x_batch_1:", x_batch_1)
#            print("t_batch_1:", t_batch_1)
#            print("x_batch_2:", x_batch_2)
#            print("t_batch_2:", t_batch_2)


