# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/6
 Description: 
"""

import random
import pandas as pd
import numpy as np
import sys


def loadTxt(file):
    data = pd.read_table(file, sep='[\s]+', engine='python', index_col=None, header=None)
    return data


def model(x, theta):
    x = np.dot(x, theta)
    1.0 / (1 + np.exp(-x))

def bgd(model, x0, niter, nfev, maxiters=10000, epsilon=1e-3):
    pass


def bgd_fit():
    pass


if __name__ == "__main__":
    data_path = r'E:\PycharmProjects\ml_impl\data\for_gd'
    data = loadTxt(data_path)