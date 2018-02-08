# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/6
 Description: 
"""

import numpy as np
import random
from math import fabs as abs


class grad_desc(object):
    """
    Batch Gradient Descent

    """

    best_thetha = None
    model = None

    @staticmethod
    def bgd_fit(model, x, y, theta, jac, alpha=0.0001, maxiters=100000):
        """
        Batch Gradient Descent
        :param model:
        :param x:
        :param y:
        :param theta:
        :param jac:
        :param alpha:
        :param maxiters:
        :return:
        """
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
        grad_desc.model = model
        return theta

    @staticmethod
    def get_cost(loss, m):
        cost = np.sum(loss ** 2) / (2.0 * m)
        return cost

    @staticmethod
    def bgd_predict(x):
        return grad_desc.model(x, grad_desc.best_thetha)

    @staticmethod
    def rgd_fit(model, x, y, theta, jac, alpha=0.00005, maxiters=100000):
        """
        Random Gradient Descent
        """
        niter = 0
        sample_idx = 0
        m, n = x.shape

        print("(m, n):(%d %d), theta0: %s" % (m, n, theta))

        while niter < maxiters:
            niter += 1
            # pick one one by one
            sample_x = x[sample_idx, :]
            sample_idx = (sample_idx + 1) % m

            prediect_y = model(sample_x, theta)
            loss = prediect_y - y[sample_idx]
            gradient = np.dot(loss, sample_x)
            theta = theta - alpha * gradient
            print("Iterations: %d | loss: %f | theta: %s" % (niter, loss, theta))

        msg = """
               RGD terminated successfully
                   loss  value: %f    
                   theta : %s
                   iterations number: %d
               """
        print(msg % (loss, theta, niter))
        grad_desc.best_thetha = theta
        grad_desc.model = model
        return theta

    @staticmethod
    def rgd_predict(x):
        return grad_desc.model(x, grad_desc.best_thetha)

    @staticmethod
    def get_batch(x, y, predict_y, batch_size=50):
        m = x.shape[0]
        sample = random.sample(range(m), batch_size)
        b_x = x[sample, :]
        b_y = y[sample]
        b_predict_y = predict_y[sample]

        # print("batch sample: (%s, %s, %s)" % (b_x, b_y, b_predict_y))
        return b_x, b_y, b_predict_y

    @staticmethod
    def mini_bgd_fit(model, x, y, theta, jac, batch_size=50, alpha=0.0001, maxiters=100000):
        niter = 0

        while niter < maxiters:
            niter += 1

            predict_y = model(x, theta)

            b_x, b_y, b_predict_y = grad_desc.get_batch(x, y, predict_y, batch_size)
            loss = b_predict_y - b_y

            # batch_size * n矩阵
            J = jac(b_x, theta)
            # n*1 的矩阵 =
            gradient = np.dot(J.T, loss) / batch_size
            theta = theta - alpha * gradient

            cost = grad_desc.get_cost(loss, batch_size)
            print("Iterations: %d | cost: %f" % (niter, cost))

        msg = """
              Mini_RGD terminated successfully
                  batch size: %d
                  loss  value: %f    
                  theta : %s
                  iterations number: %d
              """
        print(msg % (batch_size, cost, theta, niter))
        grad_desc.best_thetha = theta
        grad_desc.model = model
        return theta