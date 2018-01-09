# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:01:27 2018

@author: matsumi
"""
import matplotlib.pyplot as plt

def plot_all_top_k(train_soft_accuracies_data,train_hards_accuracies_data,train_retrievals_accuracies_data):
    # 学習曲線をプロットする
    # plot learning curves            
    plt.subplot(1, 2, 1)
    plt.title("train_soft")
    plt.plot(train_soft_accuracies_data[:,0])
    plt.plot(train_soft_accuracies_data[:,1])
    plt.plot(train_soft_accuracies_data[:,2])
    plt.plot(train_soft_accuracies_data[:,3])
    plt.legend(["train soft top-1","train soft top-2","train soft top-5","train soft top-10"], loc="best")
    plt.ylim([80, 100])
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title("train_hard")
    plt.plot(train_hards_accuracies_data[:,0])
    plt.plot(train_hards_accuracies_data[:,1])
    plt.plot(train_hards_accuracies_data[:,2])
    plt.legend(["train hard top-1","train hard top-2","train hard top-5"], loc="best")
    plt.ylim([80, 100])
    plt.grid()
        
    plt.tight_layout()
    plt.show()
    plt.draw()
    
    plt.subplot(1, 2, 1)
    plt.title("train_retrieval")
    plt.plot(train_retrievals_accuracies_data[:,0])
    plt.plot(train_retrievals_accuracies_data[:,1])
    plt.plot(train_retrievals_accuracies_data[:,2])
    plt.legend(["train retrieval top-1","train retrieval top-2","train retrieval top-5"], loc="best")
    plt.ylim([80, 100])
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    plt.draw()