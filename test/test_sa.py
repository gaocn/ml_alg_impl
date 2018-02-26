# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/26
 Description: 
"""
import numpy.random as rnd
from heuristic.sa import SimulatedAnnealing


def func(x):
    return 6 * x ** 7 + 8 * x ** 6 + 7 * x **3 + 5 * x ** 2


def cons(x):
    return 0 <= x <= 100


if __name__ == '__main__':
    print('Call Simulated Annealing')
    # 随机模拟退火算法
    x0 = rnd.random() * 100
    res = SimulatedAnnealing.rsa(x0, 0.98, func, cons=cons, max_iter=100)