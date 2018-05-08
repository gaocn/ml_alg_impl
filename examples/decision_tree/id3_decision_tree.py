# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/7
 Description:

    Decision Tree
    Pros: Computationally cheap to use, easy for  humans to understand learned results,
          missing values OK, can deal with irrelevant features
    Cons: Prone to overfitting
    Works with: Numeric values, nominal values
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import operator

"""
 ID3 Algorithm to split data
    1. we choose to split data in a way that makes unorganized data more organized.
    2. Information Gain: change in information before and after the spit.
    3. information measures: Shannon Entropy----熵值越高，数据越混乱
            Entropy = - sum{p(xi)*log(p(xi))} i in [1, n]
            p(xi)类别xi出现的概率，n为样本类别数
            
     Gini impurity：
     Introduction to Data Mining by Pan-Ning Tan, Vipin Kumar, and Michael Steinbach;
      Pearson Education (Addison-Wesley, 2005), 158
      
    4. find feature to split data in order to obtain highest information gain
    5. stop condition: 
        - run out of features on which to split or set maximum number of splits
        - all instances in a branch are same class
    Q: if we run out of features but the class labels are not all the same?
    A: Majority Vote
    
"""

##################################################################
#                  Construct   Decision  Tree                    #
##################################################################
def calc_entropy(labels):
    """
     Entropy = - sum{p(xi)*log(p(xi))} i in [1, n]
    :param labels: 类别实例，默认最后一列为类别
    :return:
    """
    n = len(labels)
    clz_count = defaultdict(int)
    for vec in labels:
        curr_label = vec[-1]
        clz_count[curr_label] += 1

    shannon_entropy = 0.0
    for label in clz_count:
        prob = float(clz_count[label])/n
        shannon_entropy -= prob * np.log2(prob)
    return shannon_entropy


def split_data_by_feature(dataset, axis, value):
    """
    split dataset on a given feature, processing one row at a time

    [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    >> split_dataset_by_feature(myDat,0,1)
    [[1, 'yes'], [1, 'yes'], [0, 'no']]
    >> split_dataset_by_feature(myDat,0,0)
    [[1, 'no'], [1, 'no']]

    :param dataset:
    :param axis: feature to split
    :param value: the value of the feature to return
    :return:
    """
    ret_data = []
    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduced_feature_vec = np.concatenate((feature_vec[:axis], feature_vec[axis+1:]), axis=0)
            ret_data.append(reduced_feature_vec)
    return np.array(ret_data)


def choose_best_feature_to_split(dataset):
    m, num_feature = dataset.shape
    init_entropy = calc_entropy(dataset)
    best_feature = -1
    best_info_gain = 0.0

    for i in range(num_feature - 1):
        features = np.unique(dataset[:, i])
        feat_entropy = 0.0
        for feature in features:
            sub_dataset = split_data_by_feature(dataset, i, feature)
            prob = len(sub_dataset) / float(m)
            feat_entropy += prob * calc_entropy(sub_dataset)
        print('feature %d entropy: %f' % (i, feat_entropy))
        info_gain = init_entropy - feat_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_vote_on_class(labels):
    """
    if we run out of features but the class labels are not all the same?
    We take majority Vote
    :param labels: numpy array shape=(m, 1)
    :return:
    """
    class_votes = defaultdict(int)
    for vote in labels:
        class_votes[vote] += 1
    sorted_class_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_votes[0][0]


def build_id3_decision_tree(dataset, feature_names):
    """
    stop condition:
        - all instances in a branch are same class
        - run out of features on which to split or set maximum number of splits
            - if we run out of features but the class labels are not all the same? using majority vote
    
    using python DICT to represent decision tree    
    
    :param dataset: data used to construct ID3 decision tree
    :param feature_names: feature names to better understand constructed tree
    :return: 
    """
    # stop condition 1
    clz_list = [row[-1] for row in dataset]
    if clz_list.count(clz_list[0]) == len(clz_list):
        return clz_list[0]

    # stop condition 2
    # 只剩下一个特征，每一次拆分会较少一个特征
    if len(dataset[0]) == 1:
        return majority_vote_on_class(dataset)

    best_feature = choose_best_feature_to_split(dataset)
    best_feat_name = feature_names[best_feature]

    decision_tree = {best_feat_name: {}}
    np.delete(feature_names, best_feature, axis=0)

    uni_feature_vals = np.unique(dataset[:, best_feature])
    for val in uni_feature_vals:
        decision_tree[best_feat_name][val] = build_id3_decision_tree(
            split_data_by_feature(dataset, best_feature, val),
            feature_names
        )
    return decision_tree


def classify(inTree, feature_names, test):
    root_feature = list(inTree.keys())[0]
    sub_tree = inTree[root_feature]
    feature_index = list(feature_names).index(root_feature)

    for key in sub_tree:
        if test[feature_index] == key:
            if type(sub_tree[key]).__name__ == 'dict':
                clz_label = classify(sub_tree[key], feature_names, test)
            else:
                clz_label = sub_tree[key]
    return clz_label


##################################################################
#                  Plot   Decision  Tree                         #
##################################################################
def get_leaf_node_num(tree):
    if type(tree).__name__ != 'dict':
        return 1
    leaf_num = 0
    for node in tree:
        leaf_num += get_leaf_node_num(tree[node])
    return leaf_num


def get_max_depth(tree):
    if type(tree).__name__ != 'dict':
        return 1
    max_depth = 0
    for node in tree:
        depth = get_max_depth(tree[node]) + 1
        max_depth = max(max_depth, depth)
    return max_depth


decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_decision_node(text, parent_pos, child_pos, node_type):
    """
     xy:       point to annotate
     xytext:   place of the text
     xycoords: The coordinate system that `xy` is given in.
     textcoords: 默认与`xycoords`一致
     arrowprops: properties used to draw a arrow between `xy` and `xytext`
    """
    create_plot.ax.annotate(text, xy=parent_pos, xytext=child_pos, xycoords='axes fraction',
                va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def plot_middle_text(parent_pos, child_pos, text):
    x_mid = parent_pos[0] + (child_pos[0] - parent_pos[0])/2.0
    y_mid = parent_pos[1] + (child_pos[1] - parent_pos[1])/2.0
    create_plot.ax.text(x_mid, y_mid, text)


def create_plot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # frameon: 图的边框
    create_plot.ax = fig.add_subplot(111, frameon=False)
    # parent_pos = (0.5, 1.0)
    # child_pos = (0.5, 1.0)
    # plot_decision_node('decision node', parent_pos, child_pos, decision_node)
    # plot_middle_text(parent_pos, child_pos, 'branch')
    # plot_decision_node('leaf node', (1.2, 1.2), (0.5, 1.0), leaf_node)
    plot_decision_tree.total_width = float(get_leaf_node_num(inTree))
    plot_decision_tree.total_height = float(get_max_depth(inTree))
    plot_decision_tree.x_off = -0.5 / plot_decision_tree.total_width
    plot_decision_tree.y_off = 1.0

    init_pos = (0.5, 1.0)
    plot_decision_tree(inTree, init_pos, '')
    plt.show()


def plot_decision_tree(tree, parent_pos, text):
    """
    in order to plot decision tree properly, we need know:
      1. `number of leaf nodes` to size things in X coordination
      2. `decision tree depth` to size things in Y coordination

    :param tree: dict to represent a decision tree
    :return:
    """
    print('plot decision tree')
    num_leaf = get_leaf_node_num(tree)
    node_text = list(tree.keys())[0]

    child_pos = (
        plot_decision_tree.x_off + (1.0 + float(num_leaf)) / 2.0 / plot_decision_tree.total_width
        ,
        plot_decision_tree.y_off
    )
    plot_decision_node(node_text, parent_pos, child_pos, decision_node)
    plot_middle_text(parent_pos, child_pos, text)

    plot_decision_tree.y_off -= 1.0 / plot_decision_tree.total_height
    for key in tree[node_text]:
        if type(tree[node_text][key]).__name__ == 'dict':
            plot_decision_tree(tree[node_text][key], child_pos, str(key))
        else:
            plot_decision_tree.x_off += 1.0 / plot_decision_tree.total_width
            plot_decision_node(tree[node_text][key], child_pos,
                               (plot_decision_tree.x_off, plot_decision_tree.y_off),
                                leaf_node)
            plot_middle_text(child_pos,
                             (plot_decision_tree.x_off, plot_decision_tree.y_off),
                             str(key))
    plot_decision_tree.y_off += 1.0 / plot_decision_tree.total_height


if __name__ == '__main__':
    data = np.array([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']])
    feature_names = np.array(['no surfacing', 'flippers'])
    print('Shannon Entropy: ', calc_entropy(data))

    print('split by feature<%s>:\n' % feature_names[0], split_data_by_feature(data, 0, '1'))
    print('split by feature<%s>:\n' % feature_names[1], split_data_by_feature(data, 0, '0'))
    print('Best feature:', choose_best_feature_to_split(data))

    tree = build_id3_decision_tree(data, feature_names)
    print('Decision Tree: \n', tree)
    leaf_num = get_leaf_node_num(tree)
    print('Decision Tree Leaf Number: \n', leaf_num)
    depth = get_max_depth(tree)
    print('Decision Tree Max Depth: \n', depth)
    # tree = {'no surfacing ': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 3:'maybe'}}
    # create_plot(tree)

    print('classify result: ', classify(tree, feature_names, ['1', '0']))
    print('classify result: ', classify(tree, feature_names, ['0', '0']))
    print('classify result: ', classify(tree, feature_names, ['0', '1']))
    print('classify result: ', classify(tree, feature_names, ['1', '1']))