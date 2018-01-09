# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:01:27 2018

@author: matsumi
"""
import matplotlib.pyplot as plt

def plot_all_top_k(soft_accuracies_data,hards_accuracies_data,retrievals_accuracies_data):
    # 学習曲線をプロットする
    # plot learning curves            
    plt.subplot(1, 2, 1)
    plt.title("soft")
    plt.plot(soft_accuracies_data[:,0])
    plt.plot(soft_accuracies_data[:,1])
    plt.plot(soft_accuracies_data[:,2])
    plt.plot(soft_accuracies_data[:,3])
    plt.legend(["soft top-1","soft top-2","soft top-5","soft top-10"], loc="best")
    plt.ylim([80, 100])
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title("hard")
    plt.plot(hards_accuracies_data[:,0])
    plt.plot(hards_accuracies_data[:,1])
    plt.plot(hards_accuracies_data[:,2])
    plt.legend(["hard top-1","hard top-2","hard top-5"], loc="best")
    plt.ylim([80, 100])
    plt.grid()
        
    plt.tight_layout()
    plt.show()
    plt.draw()
    
    plt.subplot(1, 2, 1)
    plt.title("retrieval")
    plt.plot(retrievals_accuracies_data[:,0])
    plt.plot(retrievals_accuracies_data[:,1])
    plt.plot(retrievals_accuracies_data[:,2])
    plt.legend(["retrieval top-1","retrieval top-2","retrieval top-5"], loc="best")
    plt.ylim([80, 100])
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    plt.draw()