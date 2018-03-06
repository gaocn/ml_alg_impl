# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/9
 Description: 
"""
import numpy as np
import scipy.linalg as linalg
import reprlib


class Newton(object):
    """
    求解无约束问题
        1. 牛顿算法
        2. 阻尼牛顿算法，采用非精确线性搜索确定步长因子alpha
        3. 拟牛顿算法：DFP、BFGS、Broyden、L-BFGS
    """

    ########################################################################
    # 牛顿算法实现
    ########################################################################
    @staticmethod
    def newton(x0, func, jac, hess, epsilon=1e-10, max_iter=10000):
        """
        牛顿迭代算法
        :param x0: 初始解
        :param func: 目标函数
        :param jac: 目标函数的一阶导数
        :param hess: 目标函数的二阶偏导数
        :param epsilon: 终止误差，由梯度的
        :param max_iter: 最大迭代次数
        :return: 最优解，最优解对应的函数值
        """
        niter = 0
        while niter < max_iter:
            g = jac(x0)
            H = hess(x0)
            d = -1.0 * linalg.solve(H, g)

            if linalg.norm(g) < epsilon:
                break

            x0 = x0 + d
            niter += 1
        msg = """
               Naive Newton terminated successfully
                   function value: %f    
                   x : %s
                   iterations number: %d
               """
        print(msg % (func(x0), x0, niter))
        return x0, func(x0)

    ########################################################################
    # 阻尼牛顿算法实现
    ########################################################################
    @staticmethod
    def damped_newton(x0, func, jac, hess, epsilon=1e-10, max_iter=10000, beta=0.55, delta=0.4):
        """
         阻尼牛顿算法
        :param x0: 初始解
        :param func: 目标函数
        :param jac: 目标函数的一阶导数
        :param hess: 目标函数的二阶偏导数
        :param epsilon: 终止误差，由梯度的
        :param max_iter: 最大迭代次数
        :param beta: 范围[0,1]
        :param delta: 范围[0,0.5]
        :return: 最优解，最优解对应的函数值
        """
        niter = 0
        while niter < max_iter:
            g = jac(x0)
            H = hess(x0)

            d = -1.0 * np.linalg.solve(H, g)
            # d = -1.0 * linalg.solve(H, g)
            if np.linalg.norm(d) < epsilon:
                break

            m = 0
            mk = 0

            while m < 20:
                if func(x0 + beta ** m * d) < func(x0) + delta * beta ** m * np.dot(g, d):
                    mk = m
                    print("Best alpha= %f" % (beta ** mk))
                    break
                m += 1

            x0 += beta ** mk * d
            niter += 1
            print(print("Iteration %d | func: %f" % (niter, func(x0))))

        msg = """
        Damp Newton terminated successfully
            function value: %f    
            x : %s
            iterations number: %d
        """
        print(msg % (func(x0), x0, niter))
        return x0, func(x0)


