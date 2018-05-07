# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/2
 Description:

    KNN(K-Nearest Neighbours)
    Pros： High accuracy, Insensitive to Outliers, No assumptions about data;
    Cons:  Computationally expensive, Requires a lot of memory;
    Works with: Numeric values, Nominal values(面值)

"""

"""
TARGET: determine if a movie is Romance or Action?
 
    Target Values: type of movie
    features: movie title , # of kicks, # of kisses,  

############################################################
movie title            # of kicks  # of kisses  type of movie
------------------------------------------------------------
California Man             3         104         Romance
He’s Not Really into Dudes 2         100         Romance
Beautiful Woman            1         81          Romance
Kevin Longblade            101       10          Action
Robo Slayer 3000           99        5           Action
Amped II                   98        2           Action
?                          18        90          Unknown
############################################################
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import operator


# 1. Prepare Data
def create_data_set():
    group = np.array(
        [
            [3, 104], [2, 100], [99, 5],  [98, 2],
            [1, 81],  [101, 10],
        ]
    )
    labels = np.array(['Romance', 'Romance', 'Action', 'Action', 'Romance', 'Action'])
    return group, labels


def plt_raw_data(X,y,labels):
    plt.scatter(X, y)
    # 添加标签
    for i in range(len(labels)):
        plt.text(X[i], y[i], labels[i], ha='right', wrap=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def classify1(inX, data_set, labels, k):

    diff_matrix = np.tile(inX, (len(data_set), 1)) - data_set
    square_dist = np.sum(diff_matrix ** 2, axis=1)
    sorted_dist_indices = np.argsort(square_dist)

    class_count = defaultdict(int)
    for i in range(k):
        class_count[labels[sorted_dist_indices[i]]] += 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    print(class_count)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = create_data_set()
    # plt_raw_data(group[:, 0], group[:, 1], labels)

    print('Labels:', classify1([18, 90], group, labels, 2))


