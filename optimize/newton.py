# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/9
 Description: 
"""
import numpy as np
import scipy.linalg as linalg
import scipy
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

    ########################################################################
    # 修正牛顿算法实现
    ########################################################################
    @staticmethod
    def corrected_newton(x0, func, jac, hess, epsilon=1e-10, max_iter=10000, beta=0.55, delta=0.4, tau=0):
        """
        修正牛顿算法
        :param x0: 初始解
        :param func: 目标函数
        :param jac: 目标函数的一阶导数
        :param hess: 目标函数的二阶偏导数
        :param epsilon: 终止误差，由梯度的
        :param max_iter: 最大迭代次数
        :param beta: 范围[0,1]
        :param delta: 范围[0,0.5]
        :param tau: 阻尼因子，用于修订非正定海森矩阵，A = H + mu * I，mu = ||g||^(1 + tau)
        :return: 最优解，最优解对应的函数值
        """
        niter = 0
        n = np.shape(x0)[0]

        while niter < max_iter:
            g = jac(x0)
            H = hess(x0)

            if linalg.norm(g) < epsilon:
                break

            # 若H非正定
            mu = np.power(linalg.norm(g), 1 + tau)
            A = H + mu * np.eye(n)

            d = -1.0 * linalg.solve(A, g)

            m = 0
            mk = 0
            while m < 20:
                if func(x0 + beta ** m * d) < func(x0) + delta * beta ** m * np.dot(g, d):
                    mk = m
                    break
                m += 1
            x0 = x0 + beta**mk * d
            niter += 1

        msg = """
        Corrected Newton terminated successfully
            function value: %f    
            x : %s
            iterations number: %d
        """
        print(msg % (func(x0), x0, niter))
        return x0, func(x0)

    ########################################################################
    # 拟牛顿算法实现-DFP
    ########################################################################
    @staticmethod
    def dfp(x0, func, jac, max_iter=10000, epsilon=1e-10, beta=0.55, delta=0.4):
        """
         DFP算法
        :param x0: 初始解
        :param func: 目标函数
        :param jac: 目标函数的一阶导数
        :param epsilon: 终止误差，由梯度的
        :param max_iter: 最大迭代次数
        :param beta: 范围[0,1]
        :param delta: 范围[0,0.5]
        :return:最优解，最优解对应的函数值
        """
        niter = 0
        n = np.shape(x0)[0]
        D = np.eye(n)

        while niter < max_iter:

            g = jac(x0)
            if np.linalg.norm(g) < epsilon:
                break
            d = -1.0 * np.dot(D, g)

            # 确定长因子
            m, mk = 0, 0
            while m < 20:
                if func(x0 + beta**m * d) < func(x0) + delta * beta**m * np.dot(g, d):
                    mk = m
                    break
                m += 1

            # 校正矩阵D = H^{-1}
            x = x0 + beta**mk * d
            s = x - x0
            y = jac(x) - g

            if np.dot(s, y) > 0:
                print("iteration %d | corrected D= %s，cost=%f" % (niter, D, func(x0)))
                Dy = np.dot(D, y)
                # np.dot(s, y)为常数
                # s.reshape((n, 1)) * s = np.dot(s.reshape((n, 1)), s.reshape((1, n)))  为 n * n的矩阵
                # Dy.reshape((n, 1)) * Dy 为 n * n的矩阵
                D += 1.0 * s.reshape((n, 1)) * s / np.dot(s, y) \
                     - 1.0 * Dy.reshape((n, 1)) * Dy/ np.dot(np.dot(y, D), y)

            x0 = x
            niter += 1

        msg = """
        Quasi-Newton(DFP) terminated successfully
            function value: %f    
            x : %s
            iterations number: %d
        """
        print(msg % (func(x0), x0, niter))
        return x0, func(x0)

    ########################################################################
    # 拟牛顿算法实现-BGFS
    ########################################################################
    @staticmethod
    def bgfs(x0, func, jac, max_iter=10000, epsilon=1e-10, beta=0.55, delta=0.4):
        """
        BGFS算法
        :param x0: 初始解
        :param func: 目标函数
        :param jac: 目标函数的一阶导数
        :param epsilon: 终止误差，由梯度的
        :param max_iter: 最大迭代次数
        :param beta: 范围[0,1]
        :param delta: 范围[0,0.5]
        :return:最优解，最优解对应的函数值
        :return:
        """
        niter = 0
        n = np.shape(x0)[0]
        I = np.eye(n)
        D = I

        while niter < max_iter:
            g = jac(x0)
            if np.linalg.norm(g) < epsilon:
                break

            d = -1.0 * np.dot(D, g)

            # 确定步长因子
            m, mk = 0, 0
            while m < 20:
                if func(x0 + beta**m * d) < func(x0) + delta * beta**m * np.dot(g, d):
                    mk = m
                    break
                m += 1

            x = x0 + beta**mk * d
            # 更新矩阵
            s = x - x0
            y = jac(x) - g

            if np.dot(s, y) > 0:
                print("niter=%d | best D= %s" % (niter, D))
                sy = np.dot(s, y)  # 常数
                rho = 1.0 / sy
                M1 = I - rho * scipy.outer(s, y)
                M2 = I - rho * scipy.outer(y, s)

                D = np.dot(M1, np.dot(D, M2)) + scipy.outer(s, s) * rho

            x0 = x
            niter += 1

        msg = """
        Quasi-Newton(DFP) terminated successfully
            function value: %f    
            x : %s
            iterations number: %d
        """
        print(msg % (func(x0), x0, niter))
        return x0, func(x0)