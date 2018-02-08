# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/6
 Description: 
"""

import pandas as pd
from scipy.optimize import minimize
import scipy.optimize as optimize
import numpy as np
import random
from pandas import Series, DataFrame
from optimize.grad_desc import grad_desc


def loadTxt(file):
    data = pd.read_table(file, sep=',', index_col=None, header=None)
    return data


# y = theta1x + theta2x
def model(x, theta):
    return np.dot(x, theta)


def fun_jac(x, theta):
    # temp = 1.0 / ((1 - np.exp(np.dot(x, theta))) ** 2)
    # jac[:, 0] = temp * x[0]
    # jac[:, 1] = temp * x[1]
    return x

def rgd_jac(x, theta, idx):
    # temp = 1.0 / ((1 - np.exp(np.dot(x, theta))) ** 2)
    # jac[:, 0] = temp * x[0]
    # jac[:, 1] = temp * x[1]
    return x[idx, :]

def genData(numPoints,bias,variance):
    # 生成0矩阵，shape表示矩阵的形状，参数1是行，后边是列
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=(numPoints))

    # 对x、y的0矩阵填充数值
    for i in range(0,numPoints):
        x[i][0] = 1 #第i行第1列全部等于1
        x[i][1] = i  # 第i行第2列等于i
        y[i] = (i + bias) + random.uniform(0,1) * variance # 第i行第2列等于i+bias(偏倚），再加,0-1的随机数，以及方差
    return x, y


if __name__ == "__main__":
    data_path = r'E:\PycharmProjects\ml_impl\data\for_gd'
    data = loadTxt(data_path)

    x, y = genData(100, 25, 10)  # 传入参数
    m, n = np.shape(x)  # 检查x的行列数，是否一致
    theta = np.ones(n)  # 初始化

    theta1 = grad_desc.bgd_fit(model, x, y, theta, jac=fun_jac, alpha=0.0005)
    # grad_desc.bgd_fit(model, train_x, train_y, theta, fun_jac)

    theta2 = grad_desc.rgd_fit(model, x, y, theta, jac=rgd_jac, alpha=0.00005)

    theta3 = grad_desc.mini_bgd_fit(model, x, y, theta, jac=fun_jac, alpha=0.0005)




