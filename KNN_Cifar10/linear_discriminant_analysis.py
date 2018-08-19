# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 21:12:37 2018

@author: LianTao
"""

import numpy as np

class LDA:
    def __init__(self):
        self._proj_mat = None

    def train(self, X, X_test, y, k):

        '''
        X为数据集，y为label，k为目标维数
        '''
        label_ = list(set(y))
    
        X_classify = {}
    
        for label in label_:
            X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
            X_classify[label] = X1
    
        mju = np.mean(X, axis=0)
        mju_classify = {}
    
        for label in label_:
            mju1 = np.mean(X_classify[label], axis=0)
            mju_classify[label] = mju1
    
        #St = np.dot((X - mju).T, X - mju)
        #计算类内散度矩阵
        Sw = np.zeros((len(mju), len(mju)))  
        for i in label_:
            Sw += np.dot((X_classify[i] - mju_classify[i]).T,
                         X_classify[i] - mju_classify[i])
    
        #Sb=St-Sw
        #计算类内散度矩阵
        Sb = np.zeros((len(mju), len(mju)))  
        for i in label_:
            Sb += len(X_classify[i]) * np.dot((mju_classify[i] - mju).reshape(
                (len(mju), 1)), (mju_classify[i] - mju).reshape((1, len(mju))))

        #计算Sw-1*Sb的特征值和特征矩阵
        eig_vals, eig_vecs = np.linalg.eig(
            np.linalg.inv(Sw).dot(Sb))  
    
        #提取前k个特征向量
        sorted_indices = np.argsort(eig_vals)
        topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  
        
        lowDDataMat_train = np.dot(X, topk_eig_vecs)
        lowDDataMat_test = np.dot(X_test, topk_eig_vecs)
        
        return lowDDataMat_train, lowDDataMat_test


    def project(self, X):
        return X.dot(self._proj_mat.T)
