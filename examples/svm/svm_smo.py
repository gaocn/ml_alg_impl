# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/24
 Description:

    Platt's SMO算法实现，与{svm.py}中的{simple_smo}方法的区别在于如何选择alphas？
      1. Platt采用启发式算法提升寻找最优alphas的速度；
      2. Platt的外循环用于选择第一个alpha，这一步主要用跳过那些不会发生改变的alphas。
      This alternates between single passes over the entire dataset and single passes over non-bound alphas.
      The non-bound alphas are alphas that aren’t bound at the limits 0 or C.

      3. Platt的内循环用于选择第二个alpha，选择的alpha满足：maximize the step size during optimization.
      create a global cache of error values and choose from the alphas that maximize step size or Ei-Ej.

"""
import matplotlib.pyplot as plt
import numpy as np


class SMO_OptimalStruct(object):
    """
        1. 创建用于保存中间结果的数据结构

    """

    def __init__(self, X, y, C, tolerance):
        self.X = X
        self.y = y
        self.C = C
        self.tolerance = tolerance
        self.m, self.n = X.shape

        self.alphas = np.zeros((self.m, 1))
        self.b = 0

        # 保存 error_i = hi-yi的值，用于选择第二个alpha
        # 第一列：表示error_cache是否是valid
        # 第二列：error_i的值
        self.error_cache = np.zeros((self.m, 2))

    def calc_ek(self, k):
        """
        会被频繁调用，采用内联函数
        """
        hk = float(np.dot((self.alphas * self.y).T, np.dot(self.X, self.X[k, :]))) + self.b
        error_k = hk - float(self.y[k])
        return error_k

    def update_ek(self, k):
        """
        每次得到最优alphai时更新
        :param k:
        :return:
        """
        error_k = self.calc_ek(k)
        self.error_cache[k] = [1, error_k]

    @staticmethod
    def _selectj_rand(i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0,  m))
        print('selectj_rand j={0}'.format(j))
        return j

    @staticmethod
    def clip(alpha, L, H):
        if alpha > H:
            alpha = H
        if alpha < L:
            alpha = L
        return alpha

    def selectj(self, i, error_i):
        """
        用于选择第二个alpha，通过{error_cache}存放的信息选择argmax{error_i - error_k}最大时对应的alpha值，以保证每次优化时步长最大
        :param i:
        :param error_i:
        :return:
        """
        max_k = -1
        max_delta_error = 0
        error_j = 0
        # 更新缓存的error_i
        self.error_cache[i] = [1, error_i]
        # 取出有效缓存
        valid_error_cahce_list_idx = np.nonzero(self.error_cache[:, 0])[0]

        if len(valid_error_cahce_list_idx) > 1:
            for k in valid_error_cahce_list_idx:
                if k == i: continue
                error_k = self.calc_ek(k)
                delta_error = abs(error_i - error_k)
                if delta_error > max_delta_error:
                    max_delta_error = delta_error
                    max_k = k
                    error_j = error_k
            return max_k, error_j
        else:
            j = SMO_OptimalStruct._selectj_rand(i, self.m)
            error_j = self.calc_ek(j)
        return j, error_j

    def smo_inner_loop(self, i):
        """
        :param i:
        :return: 1 if succeed find a alphaj;  0 otherwise;
        """
        error_i = self.calc_ek(i)

        if not -self.tolerance < self.y[i]*error_i < self.tolerance and 0 <= self.alphas[i] <= self.C:
            j, error_j = self.selectj(i, error_i)

            old_alphai = self.alphas[i].copy()
            old_alphaj = self.alphas[j].copy()

            if self.y[i] == self.y[j]:
                L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                H = min(self.C, self.alphas[i] + self.alphas[j])
            else:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])

            if L == H: print('L==H'); return 0

            eta = 2.0 * np.dot(self.X[i, :], self.X[j, :]) - np.dot(self.X[i, :], self.X[i, :]) \
                - np.dot(self.X[j, :], self.X[j, :])
            if eta >= 0: print('eta >= 0'); return 0
            print('eta: {0}'.format(eta))

            self.alphas[j] -= self.y[j] * (error_i - error_j) / eta
            self.alphas[j] = SMO_OptimalStruct.clip(self.alphas[j], L, H)
            self.update_ek(j)

            if abs(self.alphas[j] - old_alphaj) < 0.00001:
                print('j not moving enough')
                return 0

            self.alphas[i] += self.y[i] * self.y[j] * (old_alphaj - self.alphas[j])
            self.update_ek(i)

            b1 = self.b - error_i - self.y[i] * (self.alphas[i] - old_alphai) * np.dot(self.X[i, :], self.X[i, :]) \
                - self.y[j] * (self.alphas[j] - old_alphaj) * np.dot(self.X[i, :], self.X[j, :])

            b2 = self.b - error_j - self.y[i] * (self.alphas[i] - old_alphai) * np.dot(self.X[i, :], self.X[j, :]) \
                - self.y[j] * (self.alphas[j] - old_alphaj) * np.dot(self.X[j, :], self.X[j, :])

            if 0 < self.alphas[i] < self.C:
                self.b = b1
            elif 0 < self.alphas[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0


def SMO(X, y, C, tolerance, max_niter, k_tuple=('lin', 0)):
    """
    外循环条件：
        1. 直到最大迭代数{max_niter}
        2. 遍历所有集合后alpha pairs没有发生任何改变
    :param X:
    :param y:
    :param C:
    :param tolerance:
    :param max_niter:
    :param k_tuple:
    :return:
    """
    smo_os = SMO_OptimalStruct(X,  y, C, tolerance)
    niter = 0
    entire_set = True
    alpha_pairs_changed = 0

    while niter < max_niter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0

        if entire_set:
            for i in range(smo_os.m):
                alpha_pairs_changed += smo_os.smo_inner_loop(i)
                print('fullset, iter:{0}, i: {1}, pair changed: {2}'.format(niter, i, alpha_pairs_changed))
            niter += 1
        else:
            # goes over all the non-bound alphas
            non_bounded_alpha_idx = np.nonzero((smo_os.alphas > 0) * (smo_os.alphas < smo_os.C))[0]
            for i in non_bounded_alpha_idx:
                alpha_pairs_changed += smo_os.smo_inner_loop(i)
                print('non bounded, iter:{0}, i: {1}, pair changed: {2}'.format(niter, i, alpha_pairs_changed))
            niter += 1

        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True

        print('iteration number: {0}'.format(niter))

    # calculate w using alphas
    w = np.zeros((smo_os.n, ))
    for i in range(smo_os.m):
        if smo_os.alphas[i] == 0: continue
        w += smo_os.alphas[i] * smo_os.y[i] * smo_os.X[i, :]

    return smo_os.alphas, smo_os.b, w

if __name__ == '__main__':
    from svm import load_data, plot_decision_boundary

    X, y = load_data()
    # alphas, b, w = SMO(X, y, 0.6, 0.001, 40)
    alphas, b, w = SMO(X, y, 1.0, 0.001, 40)

    plot_decision_boundary.alphas = alphas
    plot_decision_boundary.w = w
    plot_decision_boundary.b = b
    plot_decision_boundary(X, y)
















