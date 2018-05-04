# encoding: utf-8
"""
 Author: govind
 Date: 2018/4/25
 Description: 
"""
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC
from scipy.linalg import eigh
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy


def rbf_kernel(X, gamma=1):
    sq_dist = pdist(X, 'sqeuclidean')
    mat_sq_dist = squareform(sq_dist)
    K = scipy.exp(-gamma * mat_sq_dist)
    return K


def kpca(X, kernel, n_components, gamma=1):

    # 样本数
    m = np.array(X).shape[0]
    # 样本维度
    n = np.array(X).shape[1]

    # 计算核矩阵
    K = kernel(X, gamma)

    # 中心化核矩阵
    C = np.ones((m, m)) / m
    K = K - C.dot(K) - K.dot(C) + C.dot(K).dot(C)

    # 返回特征值，和特征向量
    eigvals, eigvecs = eigh(K)

    # 返回前K个特征向量构成的矩阵
    target_eigvecs = eigvecs[:, -1:-(n_components+1):-1]

    # 调整特征值从大到小
    target_eigvals = eigvals[-1:-(n_components+1):-1]

    return target_eigvals, target_eigvecs


if __name__  == '__main__':

    X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
    y = [0, 1, 0, 1]
    K = chi2_kernel(X, gamma=.5)
    svm = SVC(kernel='precomputed').fit(K, y)
    pred = svm.predict(K)
    #
    svm2 = SVC(kernel=chi2_kernel).fit(X, y)
    pred2 = svm2.predict(X)

