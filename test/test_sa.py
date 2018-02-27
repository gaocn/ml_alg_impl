# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/26
 Description: 
"""
import numpy.random as rnd
from heuristic.sa import SimulatedAnnealing
import numpy as np

def func(*args):
    """
    :param x: 取值范围[-10, 10]
    :param y: 取值范围[-10, 10]
    :return:
    """
    x, y = args[0]
    return x ** 2 + y ** 2 + 5 * x * y - 4


def perturb(x0, y0):
    def wrapper():
        x = x0 + rnd.uniform(-10, 10)
        y = y0 + rnd.uniform(-10, 10)

        x =  10 if x >  10 else x
        x = -10 if x < -10 else x
        y =  10 if y >  10 else y
        y = -10 if y < -10 else y

        return x, y
    return wrapper


if __name__ == '__main__':
    print('Call Simulated Annealing')
    # 随机模拟退火算法
    # x0 = rnd.random() * 100
    x = rnd.uniform(-10, 10)
    y = rnd.uniform(-10, 10)
    # res = SimulatedAnnealing.rsa(x0, 0.99, func, cons=cons, max_iter=1000)
    res = SimulatedAnnealing.rsa(perturb(x, y), func, cooling_factor=0.95)

