# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/16
 Description:

    Estimate Horse Fatalities from Colic

    train data: 299  *  22
    test data: 67  *  22

   【Dealing with Missing  Values】
     1. Missing values in sample data
      We want a value that won’t impact the weight during the update.
      Set Missing Value data[rand_idx] = 0, then
        weights = weights + alpha * error * data[rand_idx]
        weights = weights
        sigmoid(0) = 0.5
      So replacing missing value with 0 will keep the imperfect data
        without compromising the learning algorithm

     2. Missing values in class label
       Given that we are using Logistic Regression, we simply threw it out.
       But for kNN, it may not make sense.

"""
import numpy as np


def load_data(file_name):
    data = np.loadtxt(file_name, delimiter='\t')
    constant_vector = np.ones((data.shape[0], 1))

    data = np.column_stack((constant_vector, data))
    return data[:, :-1], data[:, -1:]


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def stochastic_gradient_descent(X, y, max_niter=100):
    """
        w = w - alpha * data[rand_idx] *  error
    :param X:
    :param y:
    :param max_niter:
    :return:
    """
    m, n = X.shape
    w = np.zeros((n, 1))

    for i in range(max_niter):
        data_indices = list(range(m))
        for j in range(m):
            alpha = 4.0 / (i + j + 1.0) + 0.01
            rand_idx = int(np.random.uniform(0, len(data_indices)))
            h = sigmoid(np.dot(X[rand_idx, :], w))
            error = h - float(y[rand_idx])
            w = w - alpha * np.outer(X[rand_idx, :], error)
            print('{0} iterations with error {1} weight {2} alpha={3}'.format(i, error, w, alpha))
            del(data_indices[rand_idx])
    classify.w = w
    return w


def classify(X):
    result = []
    for i in X:
        h = sigmoid(np.dot(i, classify.w))
        if h > 0.5:
            result.append(1)
        else:
            result.append(0)

    result = np.reshape(result, (len(result), 1))
    return result


if __name__ == '__main__':
    train_file = r'E:\PycharmProjects\ml_impl\examples\logistic_regression\data\horseColicTraining.txt'
    test_file = r'E:\PycharmProjects\ml_impl\examples\logistic_regression\data\horseColicTest.txt'

    np.set_printoptions(suppress=True)

    train_data, train_labels = load_data(train_file)
    test_data, test_labels = load_data(test_file)

    w = stochastic_gradient_descent(train_data, train_labels)

    prod = classify(test_data)

    # error rate:35%, this wasn’t bad with over 30% of the values missing.
    error_rate = float(np.sum(prod != test_labels) / len(test_labels))
    print('error rate: {0}'.format(error_rate))

