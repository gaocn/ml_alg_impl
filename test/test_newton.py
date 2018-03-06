# encoding: utf-8
"""
 Author: govind
 Date: 2018/2/9
 Description: 
"""
import numpy as np
from  optimize.newton import Newton as nt


###################################################
# 范例：求解非约束优化问题
#      f(x) = 100(x1^2 - x2)^2 + (x1 - 1)^2
# 该问题的精确解为
#      x^* = (1,1)^T, f(x^*) = 0
#
# 梯度向量为：
#      [400(x1^2 - x2)x1 + 2(x1 - 1), -200(x1^2 - x2)]
#
# 海森矩阵为:
#       1200x1^2 - 400x2 + 2, -400x1
#       -400x1,               200
#
###################################################

def func(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2


# 梯度向量g
def jac(x):
    return np.array([400*(x[0]**2 - x[1]) * x[0] + 2*(x[0] - 1), -200*(x[0]**2 - x[1])])


# 海森矩阵 g
def hess(x):
    return np.array([
        [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
        [-400*x[0],                   200]
    ])


if __name__ == "__main__":
    print(nt)
    x0 = np.random.random(size=2)
    # res = nt.damp_newton(x0, func, jac, hess)
    res = nt.newton(x0, func, jac, hess)