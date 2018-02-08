# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/6
 Description: 
"""

import numpy as np
from math import fabs as abs


class grad_desc(object):
    """
    Batch Gradient Descent

    """

    best_thetha = None

    @staticmethod
    def bgd_fit(model, x, y, theta, jac, alpha=0.01, maxiters=100000):
        # m为样本量， n为特征值个数
        m, n = x.shape
        niter = 0

        while niter < maxiters:
            niter += 1
            predict_y = model(x, theta)
            loss = predict_y - y
            cost = grad_desc.get_cost(loss, m)
            print("Iteration %d | Cost: %f" % (niter, cost))

            # m*n的矩阵
            j = jac(x, theta)
            # n*1的向量 = n*m矩阵 dot m*1列向量
            gradient = np.dot(j.T, loss) / m
            theta = theta - alpha * gradient

        msg = """
        BGD terminated successfully
            Cost function value: %f    
            theta : %s
            iterations number: %d
        """
        print(msg % (cost, theta, niter))
        grad_desc.best_thetha = theta
        return theta

    @staticmethod
    def get_cost(loss, m):
        cost = np.sum(loss ** 2) / (2.0 * m)
        return cost
