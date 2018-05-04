# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/3
 Description: 
"""
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as  np
from collections import defaultdict
import operator


"""
TARGET: improving matches from a dating site with  kNN
FEATURES: 
          - Number of frequent flyer miles earned per year
          - Percentage of time spent playing video games
          - Liters of ice cream consumed per week
CLASS：['largeDoses', 'smallDoses', 'didntLike']


"""

features = [
    'Number of frequent flyer miles earned per year',
    'Percentage of time spent playing video games',
    'Liters of ice cream consumed per week'
]
target_name = {
    1: 'not at all',
    2: 'in small doses',
    3: 'in large doses'
}

colors = ['r', 'g', 'b']


def file2matrix(file_name):
    data = np.loadtxt(file_name, delimiter='\t')
    return data


def normlize_data(data):
    X, y = data[:, :-1], data[:, -1]
    scaler = MinMaxScaler()
    norm_x = scaler.fit_transform(X)
    return np.column_stack((norm_x, y))


def plot_rnd2features(X):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for index, label in target_name.items():
        data_x = X[X[:, -1] == index]
        ax.scatter(data_x[:, 0], data_x[:, 1], color=colors[index - 1],label=label)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.legend()
    plt.show()


def classify(inX, X, y, k):

    diff = np.tile(inX, (X.shape[0], 1)) - X
    square_sum = np.sum(diff ** 2, axis=1)
    sorted_indices = np.argsort(square_sum)

    k_nearest_neighbour = defaultdict(int)
    for i in range(k):
        k_nearest_neighbour[y[sorted_indices[i]]] += 1
    # 得到排序后的K个邻居类型列表
    sorted_k_nerest_neighbour = sorted(k_nearest_neighbour.items(),
                                        key=operator.itemgetter(1), reverse=True)
    # 取出第一个为该数据的标签
    return sorted_k_nerest_neighbour[0][0]


def dating_test(data, ratio=0.1):
    m = data.shape[0]
    num_test = int(ratio * m)
    error_count = 0
    for i in range(num_test):
        result = classify(data[i, :-1], data[num_test:, :-1], data[num_test:, -1], 3)
        print('classifier answer: %d, the real answer: %d' % (result, data[i, -1]))
        if result != data[i, -1]:
            error_count += 1
    print('total error rate is: %f' % (error_count/float(m)))


if __name__ == '__main__':
    file_name = r'data/datingTestSet2.txt'
    data = file2matrix(file_name)
    # plot_rnd2features(data)
    norm_data = normlize_data(data)
    # plot_rnd2features(norm_data)
    dating_test(norm_data)




