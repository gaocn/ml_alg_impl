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

    @staticmethod
    def rsa(solution_func, cost_func, initial_temperature=100, k=1, cooling_factor=0.98, max_iter=1000):
        """
        随机模拟退火算法
        NOTE: 一次得到的解可能是局部最优，因此使用模拟该算法是需要运行多次，找到可能的全局最优解。
        :param solution_func: 产生随机解的函数，需要能够遍布整个解空间
        :param cost_func: 求解方程的表达式
        :param initial_temperature: 初始温度
        :param k: 玻尔兹曼常数
        :param cooling_factor: 降温因子
        :param max_iter: 每一次求解最优解的最大迭代次数
        :return:
        """
        # max_temperature = 100
        min_temperature = 0.00001
        temperature = initial_temperature

        best_ans = solution_func()
        best_energy = cost_func(best_ans)

        while temperature > min_temperature:
            improved = False
            energy = None
            asn = None
            for niter in range(max_iter):
                asn = solution_func()
                energy = cost_func(asn)
                delta = energy - best_energy

                if delta < 0:
                    improved = True
                    break
                else:
                    rndp = rnd.random()
                    p = np.exp(- k * delta / temperature)
                    if p > rndp:
                        improved = True
                        print("energy fron %f to %f with probability %f > %f" % (energy, best_energy, p, rndp))
                        break
            if improved:
                best_ans = asn
                best_energy = energy

            temperature = temperature * cooling_factor

        return best_ans, best_energy