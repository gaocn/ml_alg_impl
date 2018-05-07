# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/4
 Description:
    a handwriting recognition system using KNN
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import operator
from collections import defaultdict

"""
TARGET: hand writing recognition
a        instance:  32*32 text representing a picture
training samples:   about 2000, each digit has 200 train samples
test     samples:   about 900, each digit has 90 test samples


STEPS:
  1. convert 32*32 to 1*1024, reformat images to a single vector
  2. use kNN to train and test 

"""
def img2vector(path):
    if not os.path.exists(path) and not  os.path.isdir(path):
        print('%s invalid' % path)
        return

    img_files = os.listdir(path)
    m = len(img_files)
    images = np.zeros((m, 1024))
    labels = np.zeros((m, 1))
    # TODO
    for i in range(10):
        labels[i * int(m / 10):(i + 1)*int(m / 10) - 1, 0] = i

    for row in range(m):
        img_vector = np.zeros((1, 1024))
        fd = open('%s/%s' % (path, img_files[row]))
        for i in range(32):
            line = fd.readline()
            for j in range(32):
                img_vector[0, 32*i + j] = line[j]

        images[row, :] = img_vector
    return images, labels


def classify(test, X, y, k):
    test_m = test.shape[0]
    train_m = X.shape[0]
    result = np.zeros((test_m, 1))

    for i in range(test_m):
        diff = np.tile(test[i, :], (train_m, 1)) - X
        square_sum = np.sum(diff ** 2, axis=1)
        sorted_indices = np.argsort(square_sum)

        k_nearest_neighbor = defaultdict(int)
        for j in range(k):
            k_nearest_neighbor[y[sorted_indices[j], 0]] += 1
        sorted_k_nearest_neighbour = sorted(k_nearest_neighbor.items(),
               key=operator.itemgetter(1), reverse=True)
        result[i, 0] = sorted_k_nearest_neighbour[0][0]
    return result


def hand_writing_test(test, test_labels, train, train_labels):
    pred = classify(test, train, training_labels, 7)
    error_count = np.sum(pred != test_labels)
    print('total error count: %f' % (error_count/test_labels.shape[0]))


if __name__ == '__main__':
    print('handwriting recognition system using KNN')
    training_path = r'data/training_digits'
    test_path = r'data/test_digits'
    training_img, training_labels = img2vector(training_path)
    test_img, test_labels = img2vector(test_path)

    hand_writing_test(test_img, test_labels, training_img,  training_labels)