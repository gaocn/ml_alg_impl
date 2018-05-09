# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/9
 Description:

    Pros: works with a small amount data, handles multiple classes
    Cons: sensitive to how the input data is prepared
    works with: Nominal values
"""

"""
    1. Naive Bayes is part of Bayesian decision theory
    2. Bayes is popular algorithm for document-classification problem
    
Modeling Naive Bayes
    if we need N samples for one feature, we need N^10 for 10 features 
    and N^1000 for our 1,000-feature vocabulary. 
    
    ASSUMPTIONS
     - independence among features, 即假设单词'bacon'出现在unhealthy后delicious周围的概率是一样的，实际上是不对的。
     - every feature is equal important，实际上也不是的，因为1000个单词中可能10-20个就能确定分类结果。 
    
    With assumptions, our N^1000 data get reduced to 1000*N
    
    1. How many features is that? 
      - using English Dictionary words: 500,000 as features
      - using collected to construct features
    2. convert text to token  vectors,  in which 1 represents the token existing and 0 otherwise. 
      - A TOKEN: is any combination of characters, suck URLs, IPs, words 
    
    
    P(c|w) = P(w|c)P(c)/P(w)
    Implementation Tricks
    1. multiply 0
      - Laplace Smoothing
    2. underflow, too many multiplications of small numbers
      - take natural logarithm of product, ln(a*b)=lna + lnb 
    
    Example: abusive/negative(1), not abusive(0)
     
"""
import numpy as np
from collections import defaultdict


def load_dataset():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    # 1 is abusive, 0 is not
    labels = [0, 1, 0, 1, 0, 1]
    return posting_list, labels


def create_vocabulary_list(dataset):
    vocabulary = set()
    for document in dataset:
        vocabulary |= set(document)
    return list(vocabulary)


def words2vector(vocabulary_list, words):
    words_vector = [0] * len(vocabulary_list)

    for word in words:
        if word in vocabulary_list:
            words_vector[vocabulary_list.index(word)] = 1
        else:
            print('the word {0} is not in vocabulary!'.format(word))
    return words_vector


def words2vector_bow(vocabulary_list, words):
    """
    Bag f words
    :param vocabulary_list:
    :param words:
    :return:
    """
    words_vector = [0] * len(vocabulary_list)

    for word in words:
        if word in vocabulary_list:
            words_vector[vocabulary_list.index(word)] += 1
        else:
            print('the word {0} is not in vocabulary!'.format(word))
    return words_vector


def train_NB(data, labels):
    """
    P(c|w) = P(w|c)P(c)/P(w) [分母可以忽略]
    lnP(c|w) = lnP(w|c) + ln P(c) - lnP(w) [最后一项可以忽略]
    :param data:
    :param labels:
    :return:
    """
    doc_num, word_num = data.shape
    uni_labels = np.unique(labels)
    uni_clz_count = len(uni_labels)

    # P(w|c) with Laplace Smoothing
    # prob_words_in_clz = np.zeros((uni_clz_count, word_num))
    prob_words_in_clz = np.ones((uni_clz_count, word_num))

    # P(c) with Laplace Smoothing
    # prob_clz          = np.zeros((uni_clz_count, 1))
    prob_clz          = np.ones((uni_clz_count, 1))

    #P(w) 实际比较时，可以忽略分母，简化运算
    # prob_words        = np.zeros((1, word_num))

    for i in range(len(uni_labels)):
        label_count = np.sum(labels == uni_labels[i])
        prob_clz[i, 0] = label_count / float(len(labels))
        for word_vec in data[labels == uni_labels[i]]:
            prob_words_in_clz[i, :] += word_vec

        #  using Laplace Smoothing and to prevent underflow,
        prob_words_in_clz[i, :] = np.log(
            prob_words_in_clz[i, :]/(np.sum(data[labels == uni_labels[i]] + word_num))
        )

    prob_clz = np.log(prob_clz)
    # for i in range(doc_num):
    #     prob_words[0, :] += data[i, :]
    # prob_words /= doc_num
    return prob_words_in_clz, prob_clz, uni_labels


def classify_NB(vec2classify, prob_words_clz, prob_clz, labels):
    """
    比较不同类别的lnP(c|w的大小)
    lnP(c|w) = lnP(w|c) + lnP(c)
      - if lnP(c1|w) > ln(c2|w), then return labels[0]
      - if lnP(c1|w) < ln(c2|w), then return labels[1]
    :return:
    """
    p0 = np.sum(prob_words_in_clz[0, :] * vec2classify) + prob_clz[0, 0]
    p1 = np.sum(prob_words_in_clz[1, :] * vec2classify) + prob_clz[1, 0]
    if p0 > p1:
        return labels[0]
    else:
        return labels[1]


if __name__ == '__main__':
    dataset, labels = load_dataset()

    vocabulary = create_vocabulary_list(dataset)
    print('vocabulary: ', vocabulary)

    words_vects = np.zeros((len(dataset), len(vocabulary)))
    for i in range(len(dataset)):
        # words_vects[i, :] = words2vector(vocabulary, dataset[i])
        words_vects[i, :] = words2vector_bow(vocabulary, dataset[i])
    prob_words_in_clz, prob_clz, uni_labels = train_NB(words_vects, np.array(labels))

    test_doc = ['love', 'my', 'dalmation']
    # test_vect = words2vector(vocabulary, test_doc)
    test_vect = words2vector_bow(vocabulary, test_doc)
    print('classify result: ', classify_NB(test_vect, prob_words_in_clz, prob_clz, uni_labels))

    test_doc = ['stupid', 'garbage']
    # test_vect = words2vector(vocabulary, test_doc)
    test_vect = words2vector_bow(vocabulary, test_doc)
    print('classify result: ', classify_NB(test_vect, prob_words_in_clz, prob_clz, uni_labels))
