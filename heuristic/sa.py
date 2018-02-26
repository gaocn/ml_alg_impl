# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/26
 Description:
    Simulated Annealing Implementation
"""
import numpy.random as rnd
import numpy as np
import reprlib


class SimulatedAnnealing(object):
    """
        Simulated Annealing Algorithm
            1. Random Simulated Annealing
            2. Deterministic Simulated Annealing
    """

    def __repr__(self):
        pass

    @staticmethod
    def rsa(t0, c, func, cons=None, w=None, max_iter=10000, epsilon=1e-10):
        """
        随机模拟退火算法
        :param t0: 初始温度
        :param c:  温度变化比率
        :param func: 能量计算公式
        :param cons: 温度的约束条件，例如温度变化范围在[20, 100]之间
        :param w:   权值
        :param max_iter: 最大迭代次数，默认为10000
        :param epsilon: 能量变化范围小于epsilon时，停止迭代
        :return: (c,E) = (基态时温度值，基态的最小能量值)
        """
        niter = 0
        t = t0
        min_energy = func(t0)

        while niter < max_iter:
            niter += 1
            # 在领域内产生新解
            t = t + (rnd.random() * 2 - 1) * t * c
            # 判断 t是否满足约束条件
            if not cons(t):
                continue
            new_energy = func(t)

            # 添加能量变化范围
            if abs(new_energy - min_energy) < epsilon:
                break

            if new_energy < min_energy:
                min_energy = new_energy
            else:
                delta = new_energy - min_energy
                p = np.exp(- delta / t)
                rand = rnd.random()
                if p > rand:
                    print("skip energy from %f to %f with probability %f > %f" % (min_energy, new_energy, p, rand))
                    min_energy = new_energy

        return t, min_energy


