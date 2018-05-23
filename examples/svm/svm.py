# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/16
 Description:


    Pros: Low generalization error, Computationally inexpensive, easy to interpret results
    Cons: Sensitive to tuning parameters and kernel choice, natively only handles binary classification
    Works with: Numeric values, nominal values

    SVM Implementation:  Sequential Minimal Optimization (SMO)

1. Separating data with the maximum margin
    【terms】
        @ linearly separable
        @ The line used to separate the dataset is called a separating hyperplane

    The hyperplane is our decision boundary, the farther a data point from a decision boundary, the more
    confident we are about the prediction.

    There more than one decision boundary, which  one is best? Instead of finding minimum average distance
    to decision boundary, we'd like to find the point closet to the separating hyperplane and make sure this
    point is as far away from the separating line as possible. This is known as  MARGIN, we want to have the
    greatest possible MARGIN, because if we make a  mistake or trained our classifier on limited data, we'd
    want it to be as robust as a s possible.

    The points closest to the separating hyperplane are known as SUPPORT VECTOR.
    Target: Maximum the distance from the separating hyperplane to the Support Vector.

2. Finding the maximum margin

    Assume separating hyperplane is:
        f = w^Tx+b, then support vector to hyperplane D = |w^Tx+b|/||w||

    How classifier works:
        f = 1  if w^Tx + b > 0; -1 if w^Tx + b < 0

    Why did we switch from class labels of 0 and 1 to -1 and 1? For math manageable, so we can write a single
    equation to describe the margin:
        1. label * (w^Tx+b), if label = 1, then w^Tx + b > 0, a point is far away from the separating plane on
            the positive side,
        2. label * (w^Tx+b), if label = -1, then w^Tx + b < 0a point is far away from the separating plane on
            the negative side,
    Both cases give us a large positive number if point is far away from separating plane.

    To find support vector we need to first minimum_{n} {label * (w^Tx+b)/ ||w||} to get nearest points to separating
    plane, then among there nearest points to find best w and b to maximum_{w, b} {minimum_{n} {label * (w^Tx+b)/ ||w||}}

                        maximum_{w, b} { minimum_{n} { label * (w^Tx+b)/ ||w|| } }

    If we set label*(w^Tx+b) = 1 for Support Vector, then we can minimize ||w||^{-1} to have a  solution. Not all of
    label*(w^Tx+b) will be equal to 1,  ONLY the closest values to the separating plane. For points far away from plane
    this value will be larger. So here CONSTRAINT is label * (w^Tx+b) will be 1.0 or greater.

            Target :    max_{w, b} { min_{n} { label * (w^Tx+b)/ ||w|| } }
            Subject to:    label * (w^Tx+b) >= 1

    Using Lagrange Multipliers to get a solution:

                        max_{a} { sum_{i=1-m} ai - 1/2* sum_{i,j=1-m} label_i * label_j * ai * aj * <xi, xj>}

                        a >= 0 and sum_{i=1-m}  ai * label_i = 0

    Above solution makes one assumption: the data is 100% linearly separable.

    Using slack variables: we can allow examples to be on the wrong side of decision boundary.

                        0 =< a <= C and sum_{i=1-m}  ai * label_i = 0

    The constant C controls weighting between our goal of making the margin large and  ensuring that most of the examples
      have a functional margin of at least 1.0.

    Once we solve for our alphas, we can write the separating hyperplane in terms of these alphas.The majority of the
    work in SVMs is finding the alphas.
                            w =  \sum_{i=1-m} ai * yi * xi


3. Platt’s SMO algorithm

    The SMO algorithm works to find a set of alphas and b. it chooses two alphas to optimize on each cycle. Once a
    suitable pair of alphas is found, one is increased and one is decreased. A set of alphas must meet certain criteria:
        1. both of the alphas have to be outside their margin boundary.
        2. the alphas aren’t already clamped or bounded.



"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def load_data():
    file_name = r'E:\PycharmProjects\ml_impl\examples\svm\data\test_set.txt'
    data = np.loadtxt(file_name, delimiter='\t')

    return data[:, :-1], data[:, -1:]


def select_jrand(i, m):
    """
        randomly selects one integer from a range
    :param i: index of first alpha
    :param m: total number of alphas
    :return:
    """
    print('select random j')
    j = i

    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_aj(aj, H, L):
    """
         clip alphas if they get too big.
    """
    if aj > H:
        aj = H
    if aj < L:
        aj = L

    print('clip_aj {0}'.format(aj))
    return aj


def simple_smo(X, y, C, tolerance, max_niter):
    """
        没有外层循环的SMO简单实现

    :param X:
    :param y:
    :param C: 松弛因子
    :param tolerance:
    :param max_niter:
    :return:
    """
    m, n = X.shape
    b = 0
    alphas = np.zeros((m, 1))

    niter = 0
    while niter < max_niter:
        # record if the attempt to optimize any alphas worked
        alphaPairsChanged = 0

        for i in range(m):
            # y = w^Tx+b
            # w =  \sum_{i=1-m} ai * yi * xi
            pred_i = float(np.dot((alphas*y).T, np.dot(X, X[i, :])) + b)
            error_i = pred_i - float(y[i])

            print('pred:{0}, error: {1}'.format(pred_i, error_i))

            if not -tolerance < y[i] * error_i < tolerance and 0 <= alphas[i] <= C:
                j = select_jrand(i, m)
                pred_j = float(np.dot((alphas*y).T, np.dot(X, X[j, :])) + b)
                error_j = pred_j - float(y[j])

                old_alphai = alphas[i].copy()
                old_alphaj = alphas[j].copy()

                # guarantee alphaj stay between 0 and C
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])

                if L == H: print('L == H'); continue

                # eta is the optimal amount to change alpha[j]
                # to simplify SMO algorithm, actually eta=0 seldom happens, so it'd ok to skip this
                eta = 2 * np.dot(X[i, :], X[j, :]) \
                      - np.dot(X[i, :], X[i, :]) \
                      - np.dot(X[j, :], X[j, :])
                if eta >= 0: print('eta >= 0'); continue
                print('eta: {0}'.format(eta))

                # 参见 Platt's SMO
                alphas[j] -= y[j] * (error_i - error_j) / eta
                alphas[j] = clip_aj(alphas[j], H, L)

                if abs(alphas[j] - old_alphaj) < 0.00001:
                    print('j not move enough')
                    continue
                # 计算alphai
                alphas[i] += y[i] * y[j] * (old_alphaj - alphas[j])

                # 计算b
                print('caculate b')

                b1 = b - error_i - y[i] * (alphas[i] - old_alphai) * np.dot(X[i, :], X[i, :])\
                     - y[j] * (alphas[j] - old_alphaj) * np.dot(X[i, :], X[j, :])

                b2 = b - error_j - y[i] * (alphas[i] - old_alphai) * np.dot(X[i, :], X[j, :])\
                     - y[j] * (alphas[j] - old_alphaj) * np.dot(X[j, :], X[j, :])

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print('{0} iterations paris changed {1}'.format(niter, alphaPairsChanged))
        if alphaPairsChanged == 0:
            niter += 1
        else:
            # alphas have been updated
            niter = 0
        print('iteration number {0}'.format(niter))
    return alphas, b


def plot_decision_boundary(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in np.unique(y):
        Xi = X[np.ravel(y == i)]
        ax.scatter(Xi[:, 0], Xi[:, 1])

    X1 = np.linspace(-1, 10, 50)
    X2 = (- plot_decision_boundary.w[0] * X1 - plot_decision_boundary.b) / plot_decision_boundary.w[1]
    ax.plot(X1, X2)

    plt.title('Support Vector Circled')
    svX = X[np.ravel(plot_decision_boundary.alphas>0)]
    svY = y[np.ravel(plot_decision_boundary.alphas>0)]

    for i in range(len(svY)):
        c = Circle((svX[i, 0], svX[i, 1]), radius=0.5, facecolor='none', alpha=0.5, edgecolor=(0, 0.8, 0.8), linewidth=3)
        ax.add_patch(c)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()


if __name__ == '__main__':
    X, y = load_data()

    alphas, b = simple_smo(X, y, 0.6, 0.001, 40)

    w = np.zeros((X.shape[1]))
    for i in range(len(alphas)):
        if alphas[i] == 0: continue
        w += alphas[i] * y[i] * X[i, :]

    plot_decision_boundary.alphas = alphas
    plot_decision_boundary.w = w
    plot_decision_boundary.b = b
    plot_decision_boundary(X, y)


















