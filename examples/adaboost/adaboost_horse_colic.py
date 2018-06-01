# encoding: utf-8
"""
 Author: govind
 Date: 2018/6/1
 Description: 
"""
import numpy as np


def load_data():
    train_file_name = r'E:\PycharmProjects\ml_impl\examples\adaboost\data\horse_colic_training2.txt'
    test_file_name = r'E:\PycharmProjects\ml_impl\examples\adaboost\data\horse_colic_test2.txt'

    data = np.loadtxt(train_file_name, delimiter='\t')
    trainX, trainY = data[:, :-1], data[:, -1:]

    data = np.loadtxt(test_file_name, delimiter='\t')
    testX, testY = data[:, :-1], data[:, -1:]
    return trainX, trainY, testX, testY


def stump_classify(X, dim, thresh, ineq):
    clz = np.ones((X.shape[0], 1))

    if ineq == 'lt':
        clz[X[:, dim] <= thresh] = -1
    else:
        clz[X[:, dim] > thresh] = -1

    return clz


def build_stump(X, y, D):
    m, n = X.shape
    num_step = 10
    min_error = np.inf
    best_stump = {}
    best_clz_pred = np.zeros((m, 1))

    for i in range(n):
        min = np.min(X[:, i])
        max = np.max(X[:, i])
        step_size = (max - min) / float(num_step)
        for j in range(-1, int(num_step)+1):
            thresh = min + int(step_size) * j
            for ineq in ['lt', 'gt']:
                pred = stump_classify(X, i, thresh, ineq)

                error = np.ones((m, 1))
                error[pred == y] = 0
                print('predicted vals: {0}, error predicted in position: {1}'.format(pred.T, error.T))

                weighted_error = np.dot(error.T, D)
                print('split dim {0}, thresh {1}, ineq {2}, weighted error {3}'.format(
                    i, thresh, ineq, weighted_error
                ))

                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh
                    best_stump['ineq'] = ineq
                    best_clz_pred = pred.copy()

        return best_stump, best_clz_pred, min_error


def adaboost(X, y, max_niter=40):
    m, n = X.shape

    D = np.ones((m, 1)) / m
    classifiers = []
    agg_pred = np.zeros((m, 1))

    for i in range(max_niter):
        stump, pred, error = build_stump(X, y, D)

        # 如果预测错误率大于50%，则跳过此分类器
        error_rate = np.sum(pred != y) / float(m)
        if error_rate > 0.5: continue

        # weight of classifier alpha = 1/2 ln[(1-error)/(error)]
        alpha = 0.5 * np.log((1 - error)/max(error, 1e-16))

        stump['alpha'] = alpha
        classifiers.append(stump)
        print('classsifier {0} added'.format(stump))

        # update weight: Di = Di * exp(-alpha * yi * hi) / sum(D)
        expon = np.exp(-1.0 * pred * y)
        D = D * expon
        D = D / np.sum(D)

        # 计算组合结果
        agg_pred += pred * D
        agg_pred_error = np.sum(np.sign(agg_pred) != y) / m
        print('agg_pred: {0}, agg_pred_error: {1}'.format(agg_pred.T, agg_pred_error.T))
        if agg_pred_error == 0: break
    return classifiers


def adaboost_classify(testX, classifiers):
    agg_pred = np.zeros((testX.shape[0], 1))

    for classifier in classifiers:
        clz_pred = stump_classify(testX, classifier['dim'], classifier['thresh'], classifier['ineq'])
        agg_pred += classifier['alpha'] * clz_pred
    return np.sign(agg_pred)


if __name__ == '__main__':
    # np.set_printoptions(suppress=True)
    X, y, testX, testY = load_data()

    # D = np.ones((X.shape[0], 1)) / X.shape[0]
    # ret = build_stump(X, y, D)

    classifiers = adaboost(X, y, 10)
    pred = adaboost_classify(testX, classifiers)

    error_rate = np.sum(pred != testY) / testX.shape[0]
    print('error rate: %.3f' % error_rate)