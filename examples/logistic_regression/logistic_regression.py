# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/14
 Description:

    Pros: Computationally inexpensive, easy to implement, knowledge representation easy to interpret
    Cons: Prone to underfitting, may have low accuracy
    Works with: Numeric values, nominal values
"""
import numpy as np


"""
    modelling a Equation  using  features and it will predict a class for a given input 

    1. logistic regression =  (f = w^Tx) + Step Function(Sigmoid Function)
    2. how to plot the decision boundary generated with gradient ascent.
    3. stochastic gradient ascent 
            f'(w) = x
    
    
"""


def load_data(file_name):
    data = np.loadtxt(file_name, delimiter='\t')
    X = data[:, 0:-1]
    y = data[:, -1:]
    return X, y


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def gradient_decent(X, y, alpha=0.001, max_niter=3000, epsilon=1e-6):
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






if __name__ == '__main__':
    print('Logistic Regression')

    file_name = r'E:\PycharmProjects\ml_impl\examples\logistic_regression\data\testSet.txt'

    X, y = load_data(file_name)

    weight = gradient_decent(X, y)