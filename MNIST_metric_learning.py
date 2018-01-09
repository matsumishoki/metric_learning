# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:15:01 2017

@author: matsumi
"""
import evaluate as e
import make_data_perm as mdp
import load_mnist
import plot_all_top_k as plot
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

if __name__ == '__main__':
    X_train, T_train, X_test, T_test = load_mnist.load_mnist()
    T_train = T_train.astype(np.int32)
    T_test = T_test.astype(np.int32)

    
     # 60000ある訓練データセットを54000と6000の評価のデータセットに分割する
    X_train, X_valid, T_train, T_valid = train_test_split(
        X_train, T_train, test_size=0.1, random_state=100)
    num_train = len(X_train)
    num_valid = len(X_valid)
    num_test = len(X_test)

    classes = np.unique(T_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = X_train.shape[-1]  # xの次元

    # 超パラメータの定義
    learning_rate = 0.0001  # learning_rate(学習率)を定義する
    max_iteration = 200      # 学習させる回数
    batch_size = 300       # ミニバッチ1つあたりのサンプル数
 
    model = M.ConvNet().to_gpu()

    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    # 訓練データのtop-kを格納する空のリストを定義する
    train_softs_accuracies=[]
    train_hards_accuracies=[]
    train_retrievals_accuracies=[]
    
    # 検証データのtop-kを格納する空のリストを定義する
    valid_softs_accuracies=[]
    valid_hards_accuracies=[]
    valid_retrievals_accuracies=[]
    
    loss_train_history = []
    loss_valid_history = []

    valid_loss_best = 10
    num_batches = num_train // batch_size  # ミニバッチの個数
    extract_size =100 # 抽出するデータ量を定義する
    num_valid_batches = num_valid // batch_size
    num_test_batches = num_test // batch_size
    i = 0
    # 学習させるループ
    for epoch in range(max_iteration):
        print ("epoch:", epoch)
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
#        print ("time_elapsed:", time_elapsed)
                       
        # 訓練データセットの交差エントロピー誤差を表示する
        train_loss = M.metric_loss_average(
                model, X_train, T_train, num_batches, False)
        loss_train_history.append(train_loss)
        print ("[train] Loss:", train_loss)
        
        # 訓練データからtop_kを求める
        train_softs_accuracy, train_hard_accuracy, train_retrieval_accuracy = e.compute_soft_hard_retrieval_from_data(X_train, T_train, extract_size, model)
        softs_K = [1,2,5,10]
        train_softs_accuracies.append(train_softs_accuracy)
        train_soft_accuracies_data = np.array(train_softs_accuracies).reshape(epoch+1, len(softs_K))
        hards_K = [2,3,4]
        train_hards_accuracies.append(train_hard_accuracy)
        train_hards_accuracies_data = np.array(train_hards_accuracies).reshape(epoch+1, len(hards_K))
        retrievals_K = [2,3,4]
        train_retrievals_accuracies.append(train_retrieval_accuracy)
        train_retrievals_accuracies_data = np.array(train_retrievals_accuracies).reshape(epoch+1, len(retrievals_K))

        
        # 検証用データセットの交差エントロピー誤差を表示する
        valid_loss = M.metric_loss_average(
                model, X_valid, T_valid, num_valid_batches, False)
        loss_valid_history.append(valid_loss)
        print ("[valid] Loss:", valid_loss)


        # 検証データの誤差が良ければwの最善値を保存する
        if valid_loss <= valid_loss_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            valid_loss_best = valid_loss
            print ("epoch_best:", epoch_best)
            print ("valid_loss_best:", valid_loss_best)
        # 検証データからtop_kを求める
        valid_softs_accuracy, valid_hard_accuracy, valid_retrieval_accuracy = e.compute_soft_hard_retrieval_from_data(X_valid, T_valid, extract_size, model_best)
        softs_K = [1,2,5,10]
        valid_softs_accuracies.append(valid_softs_accuracy)
        valid_soft_accuracies_data = np.array(valid_softs_accuracies).reshape(epoch+1, len(softs_K))
        hards_K = [2,3,4]
        valid_hards_accuracies.append(valid_hard_accuracy)
        valid_hards_accuracies_data = np.array(valid_hards_accuracies).reshape(epoch+1, len(hards_K))
        retrievals_K = [2,3,4]
        valid_retrievals_accuracies.append(valid_retrieval_accuracy)
        valid_retrievals_accuracies_data = np.array(valid_retrievals_accuracies).reshape(epoch+1, len(retrievals_K))
        
        # lossの曲線をプロットする
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
        print("train curves")
        plot.plot_all_top_k(train_soft_accuracies_data,train_hards_accuracies_data,train_retrievals_accuracies_data)
        print("valid curves")
        plot.plot_all_top_k(valid_soft_accuracies_data,valid_hards_accuracies_data,valid_retrievals_accuracies_data)
            
    # テストデータセットの交差エントロピー誤差を表示する
    test_loss = M.metric_loss_average(
            model_best, X_test, T_test, num_test_batches, False)
    print ("[valid] Loss (best):", valid_loss_best)
    print ("[test] Loss:", test_loss)
    print ("Best epoch:", epoch_best)
    print ("Finish epoch:", epoch)
    print ("Batch size:", batch_size)
    print ("Learning rate:", learning_rate)