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
import math

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
    num_test_batches = num_test / batch_size
    i = 0
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
        perm = np.random.permutation(num_train)
        
        for batch_indexes in np.split(perm, num_batches):
            x_batch = cuda.to_gpu(X_train[batch_indexes])
            t_batch = cuda.to_gpu(T_train[batch_indexes])
            
            # 勾配を初期化する            
            model.zerograds()
            
            # 順伝播させる
            y_batch = model.__call__(x_batch, True)
            # contrastive lossに入力するy1,y2,tを取得する
            y1, y2 = F.split_axis(y_batch,2, axis=0)
            t1, t2 = F.split_axis(t_batch,2, axis=0)
            t = (t1.array==t2.array).astype(np.int32)        
            loss = F.contrastive(y1, y2, t)
#            print("loss:", loss)
            # 逆伝播
            loss.backward()
            optimizer.update()
        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print ("time_elapsed:", time_elapsed)
                       
        # 訓練データセットの交差エントロピー誤差を表示する
        train_loss = M.metric_loss_average(
                model, X_train, T_train, num_batches, False)
        loss_train_history.append(train_loss)
        print ("[train] Loss:", train_loss)
        
        # 訓練データをX_trainからY_trainに変換する
        Y_train = []
        make_train_data_perm = np.random.permutation(num_train)
        with chainer.no_backprop_mode():
            for make in make_train_data_perm:
                x_train_data = cuda.to_gpu(X_train[make])
                t_train_data = cuda.to_gpu(T_train[make])
                
                y_train_data = model.__call__(x_train_data, False)
                Y_train.append(y_train_data.array)
                print('i',i)
                i = i + 1

        # Yから距離行列Dに変換する

        # 検証用データセットの交差エントロピー誤差を表示する
        valid_loss = M.metric_loss_average(
                model, X_valid, T_valid, num_valid_batches, False)
        loss_valid_history.append(valid_loss)
        print ("[valid] Loss:", valid_loss)
        
        # 学習曲線をプロットする
        # plot learning curves
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(loss_train_history)
        plt.plot(loss_valid_history)
        plt.legend(["train", "valid"], loc="best")
        plt.ylim([0.0, 0.02])
        plt.grid()
        
        plt.tight_layout()
        plt.show()
        plt.draw()

        # 検証データの誤差が良ければwの最善値を保存する
        if valid_loss <= valid_loss_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            valid_loss_best = valid_loss
            print ("epoch_best:", epoch_best)
            print ("valid_loss_best:", valid_loss_best)
            
    # テストデータセットの交差エントロピー誤差を表示する
    test_loss = M.metric_loss_average(
            model_best, X_test, T_test, num_test_batches, False)
    print ("[valid] Loss (best):", valid_loss_best)
    print ("[test] Loss:", test_loss)
    print ("Best epoch:", epoch_best)
    print ("Finish epoch:", epoch)
    print ("Batch size:", batch_size)
    print ("Learning rate:", learning_rate)