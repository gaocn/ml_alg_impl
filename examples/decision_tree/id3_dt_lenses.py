# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/8
 Description:
    Data: ['age', 'prescript', 'astigmatic', 'tearRate']
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import operator


def load_data(file_name):
    data = np.loadtxt(file_name, delimiter='\t', dtype=np.object)
    return data


def calc_entropy(data):
    shannon_entropy = 0.0
    m = data.shape[0]
    uni_clz = np.unique(data[:, -1])
    for clz in uni_clz:
        p = len(data[data[:, -1] == clz]) / float(m)
        shannon_entropy -= p * np.log2(p)
    return shannon_entropy


def split_data_by_feature(data, axis, feature_val):
    result_data = []
    for vect in data:
        if vect[axis] == feature_val:
            result_data.append(
                np.concatenate((vect[:axis], vect[axis+1:]))
            )
    return np.array(result_data)


def choose_best_feature_to_slpit(data):
    best_feature = -1
    best_info_gain = 0.0
    m, n = data.shape

    base_entropy = calc_entropy(data)

    for i in range(n - 1):
        feature_vals = np.unique(data[:, i])
        feature_enptropy = 0.0
        for feature_val in feature_vals:
            sub_data = split_data_by_feature(data, i, feature_val)
            feature_enptropy += len(sub_data)/float(m) * calc_entropy(sub_data)

        info_gain = base_entropy - feature_enptropy
        print('information gain for feature %d is: %f' % (i, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_vote(date):
    clz = np.unique(data[:, -1])
    clz_count = defaultdict(int)

    for i in clz:
        clz_count[i] += 1

    sorted_clz_count = sorted(clz_count.items(),
                               key=operator.itemgetter(1),
                               reverse=True)
    return sorted_clz_count[0][0]


def build_id3_decision_tree(data, feature_names):
    # stop condition 1
    if np.sum(data[:, -1] == data[:, -1][0]) == len(data):
        return data[:, -1][0]

    # stop condition 2
    # run out of attribute
    if len(data[0]) == 1:
        return majority_vote(data)

    best_feature = choose_best_feature_to_slpit(data)
    feature_vals = np.unique(data[:, best_feature])
    best_feature_name = feature_names[best_feature]
    tree = {best_feature_name: {}}

    np.delete(feature_names, best_feature, axis=0)

    print('feature_vals: ', feature_vals)
    print('best_feature_name: ', best_feature_name)
    print('sub_best_feature_names: ', feature_names)

    for feature_val in feature_vals:
        tree[best_feature_name][feature_val] = build_id3_decision_tree(
            split_data_by_feature(data, best_feature, feature_val)
            ,
            feature_names
        )
    return tree


if __name__ == '__main__':
    file_name = r'E:\PycharmProjects\ml_impl\examples\decision_tree\data\lenses.txt'
    feature_names = np.array(['age', 'prescript', 'astigmatic', 'tearRate'])
    data = load_data(file_name)

    entropy = calc_entropy(data)
    print('Base Entropy: ', entropy)

    split_data_test1 = split_data_by_feature(data, 1, 'hyper')
    print('Split data by feature <hyper>\n', split_data_test1)
    split_data_test2 = split_data_by_feature(data, 3, 'normal')
    print('Split data by feature <normal>\n', split_data_test2)

    best_feat = choose_best_feature_to_slpit(data)
    print('Best feature index: ', best_feat)

    tree = build_id3_decision_tree(data, feature_names)
    print("ID3 Decision Tree: \n", tree)

    from id3_decision_tree import create_plot
    create_plot(tree)