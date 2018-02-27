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
    def rsa(solution_func, cost_func, initial_temperature=10, cooling_factor=0.98, max_iter=100):
        """
        随机模拟退火算法
        :param solution_func: 产生随机解的函数，需要能够遍布整个解空间
        :param cost_func: 求解方程的表达式
        :param initial_temperature: 初始温度
        :param cooling_factor: 降温因子
        :param max_iter: 每一次求解最优解的最大迭代次数
        :return:
        """
        # max_temperature = 100
        min_temperature = 0.001
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
                    p = np.exp(- delta / temperature)
                    if p > rndp:
                        improved = True
                        print("energy fron %f to %f with probability %f > %f" % (energy, best_energy, p, rndp))
                        break
            if improved:
                best_ans = asn
                best_energy = energy

            temperature = temperature * cooling_factor

        return best_ans, best_energy