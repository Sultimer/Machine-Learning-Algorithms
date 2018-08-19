# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 07:48:28 2018

@author: LianTao
"""

import numpy as np

class PCA:
    def __init__(self, dimension = None):
        self.dimension = dimension
    
    def reduce_dimension(self, X_train, X_test):
    
        dim = X_train.shape[0]
        #将样本中心化
        x_vals = np.mean(X_train, axis=0)
        x_train_bar = X_train - x_vals
        x_test_bar = X_test - x_vals
        
        #求协方差矩阵
        x_bar_T = np.transpose(x_train_bar)
        Cov = np.dot(x_bar_T, x_train_bar) / (dim - 1)
        
        #求矩阵的特征值、特征向量
        eigVals, eigVects = np.linalg.eig(Cov)
        
        #按照特征值大小降序排列特征向量
        eigValInd = np.argsort(-eigVals)
    
        #利用方差贡献度自动选择降维维度
        varience_sum = sum(eigVals)
        varience = eigVals[eigValInd]                   
        varience_radio = varience / varience_sum
        
        varience_contribution = 0
        for newDim in range(dim):
            varience_contribution += varience_radio[newDim]
            if varience_contribution >= 0.99:
                break
        
        #也可以指定降维维数
#        newDim = 300
        
        #取前newDim个最大特征值对应的特征向量
        eigValInd = eigValInd[0: newDim+1]
        redEigVects = eigVects[:, eigValInd]
        lowDDataMat_train = np.dot(x_train_bar, redEigVects)
        lowDDataMat_test = np.dot(x_test_bar, redEigVects)
        
        return lowDDataMat_train, lowDDataMat_test
        
        #复原降维后图像
#        reconMat = np.dot(lowDDataMat_train, np.transpose(redEigVects)) + x_vals
#        return reconMat