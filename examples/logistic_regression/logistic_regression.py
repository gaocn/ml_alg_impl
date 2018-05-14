# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/14
 Description:

    Pros: Computationally inexpensive, easy to implement, knowledge representation easy to interpret
    Cons: Prone to underfitting, may have low accuracy
    Works with: Numeric values, nominal values
"""
import matplotlib.pyplot as plt
import numpy as np

"""
    modelling a Equation  using  features and it will predict a class for a given input 

    1. logistic regression =  (f = w^Tx) + Step Function(Sigmoid Function)
    2. how to plot the decision boundary generated with gradient ascent.
    3. stochastic gradient ascent 
            f'(w) = x
    
    
"""


def load_data():
    file_name = r'E:\PycharmProjects\ml_impl\examples\logistic_regression\data\testSet.txt'
    data = np.loadtxt(file_name, delimiter='\t')
    X = data[:, 0:-1]

    # 常数项恒为1 y = w0 + w1x1 +w2x2
    m = X.shape[0]
    ones = np.ones((m, 1))
    X = np.column_stack((ones, X))

    y = data[:, -1:]
    return X, y


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def gradient_decent(X, y, alpha=0.001, max_niter=3000, epsilon=1e-10):
    m, n = X.shape
    w = np.ones((n, 1))

    niter = 0
    while niter < max_niter:
        h = sigmoid(np.dot(X, w))
        error = h - y
        new_w = w - alpha * np.dot(X.T, error)
        if np.linalg.norm(new_w - w) < epsilon:
            break
        w = new_w
        print('{0} iteration, weight={1}'.format(niter, w))
        niter += 1
    return w


def plot_decision_boundry(weight):
    X, y = load_data()

    ravel_y = y.ravel()

    X0 = X[ravel_y == 0]
    X1 = X[ravel_y == 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X0[:, 1], X0[:, 2], s=30, c='red', marker='s')
    ax.scatter(X1[:, 1], X1[:, 2], s=30, c='green')

    lineX = np.linspace(-3.0, 3.0, 60)

    # decision line: 0 = w0 + w1x1 +w2x2
    lineY = (-weight[0] - weight[1] * lineX)/weight[2]
    ax.plot(lineX, lineY)

    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.show()


if __name__ == '__main__':
    print('Logistic Regression')
    X, y = load_data()

    weight = gradient_decent(X, y)

    plot_decision_boundry(weight)