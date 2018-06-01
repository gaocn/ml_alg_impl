# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/29
 Description:

    Pros: Low generation error, easy to code, works with most classifiers,
        no parameters to adjust
    Cons: Sensitive to outliers
    Works with: Numeric values, Nominal values

    Ensemble Methods or Meta-Algorithms: can take the form of using different
    algorithms, using the same algorithm with different settings, or assigning
    different parts of the dataset to different classifiers.


    Bagging (Bootstrap aggregating)
    1. The data is taken from the original dataset S times with replacement to make S new datasets,
        the datasets are the same size as the original.
    "With repalcement" means you can select the same example more than one once, which may result
    repeated examples and missing some examples.

    2. After S datasets are built, a learning algorithm is applied, so you get S classifiers. To
    predict new data, you apply S classifiers to it and TAKE a Majority Vote.

    Boosting
    In boosting, different classifiers are trained sequentially, each new classifier is trained
    based on the performance of those already trained. Boosting makes new classifier focus on data
    that was previously misclassified by previous classifier.
    Output is calculated from a weighted sum of all classifiers.

    Can we take a weak classifier and use multiple instances of it to create a strong classifier?
    "Weak classifier": the classifier does a better job than randomly guessing but not by much.
    "Strong classifier": the classifier have a much lower error rate.

    Adaptive Boosting
    1. inital weight vector D with even weight
    2. a weak classifier is trained on the data
    3. error = (number of incorrectly classified examples)/(total number of examples)
    4. weight of a classifier is given: alpha = 1/2 ln[(1-error)/(error)]
    5. update D,
            Di = Di * exp(-alpha)/sum(D)   if correctly predicted
            Di = Di * exp(alpha)/sum(D)    if incorrectly predicted
    6. jump to 2 until error=0 or number of weak classifiers reaches a user-defined value
"""
import numpy as np


def load_sample():
    X = np.array(
        [[1.,  2.1],
         [2.,  1.1],
         [1.3,  1.],
         [1.,  1.],
         [2.,  1.]]
    )
    y = np.array([[1.0, 1.0, -1.0, -1.0, 1.0]])
    return X, y.T


def stump_classify(X, dim, threshIneq, threshold):
    """
    a decision stump  makes a decision on one feature only, it's a tree with only one split.
    决策树桩(一刀切),也称单层决策树
    :return:
    """
    ret = np.ones((X.shape[0], 1))
    if threshIneq == 'lt':
        ret[X[:, dim] <= threshold] = -1.0
    else:
        ret[X[:, dim] > threshold] = -1.0
    return ret


def build_stump(X, y, D):
    """

    """
    m, n = X.shape

    num_step = 10
    best_stump = {}
    best_class_pred = np.zeros((m, 1))
    min_error = np.inf

    for col in range(n):
        min = X[:, col].min()
        max = X[:, col].max()
        step_size = (max - min) / num_step
        for j in range(-1, int(num_step) +1):
            for inequal in ['lt', 'gt']:
                thresh_val = (min + float(j)*step_size)
                pred_val = stump_classify(X, col, inequal, thresh_val)

                error = np.ones((m, 1))
                error[pred_val == y] = 0
                print('predicted vals: {0}, error_pos: {1}'.format(pred_val.T, error.T))

                weighted_error = np.dot(D.T, error)
                print('Split dim %d, thresh %.2f, thresh inequal: %s, '
                      'weighted error is: %.3f' % (col, thresh_val, inequal, weighted_error))

                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_pred = pred_val.copy()
                    best_stump['dim'] = col
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_pred


def adaboost(X, y, max_niter=40):
    m, n = X.shape
    # 样本初始权重
    D = np.ones((m, 1)) / m

    weak_clz = []
    agg_clz_pred = np.zeros((m, 1))

    for i in range(max_niter):
        best_stump, error, clz_pred = build_stump(X, y, D)
        print('samples weight D: {0}, error: {1}'.format(D.T, error))

        # weight of classifier alpha = 1/2 ln[(1-error)/(error)]
        # max：防止除数为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-12)))
        print('alpha: {0}'.format(alpha))
        best_stump['alpha'] = alpha
        weak_clz.append(best_stump)
        print('labels pred: ', clz_pred.T)

        # update D,
        #     Di = Di * exp(-alpha)/sum(D)   if correctly predicted
        #     Di = Di * exp(alpha)/sum(D)    if incorrectly predicted
        #     Di = Di * exp(-alpha * yi * hi) / sum(D)
        expon = np.exp((-1.0 * y) * clz_pred)
        D = expon * D
        D = D / np.sum(D)

        # 多分类器聚合结果
        agg_clz_pred += alpha * clz_pred
        print('agg_clz_pred:{0}'.format(agg_clz_pred.T))

        agg_error = np.sum(np.sign(agg_clz_pred) != y) / float(m)
        print('error rate: %.3f' % agg_error)

        # no error, break loop
        if agg_error == 0.0: break

    return weak_clz


def adaboost_classify(testX, classifier_arr):
    m, n = testX.shape
    agg_clz = np.zeros((m, 1))

    for classifier in classifier_arr:
        clz_pred = stump_classify(testX, classifier['dim'], classifier['ineq'], classifier['thresh'])
        agg_clz += classifier['alpha'] * clz_pred

    return np.sign(agg_clz)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    X, y = load_sample()

    # D = np.ones((X.shape[0], 1)) / 5
    # tree = build_stump(X, y, D)

    res = adaboost(X, y)

    testX = np.array([[0., 0.], [1., 1.6]])
    clz_pred = adaboost_classify(testX, res)
    print(clz_pred)



















