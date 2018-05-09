# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/9
 Description:

"""
import numpy as np
import os
import re


def parse_txt2token(ham_dir, spam_dir):
    words_list = []
    labels = []

    pattern = r'\W+'
    reg = re.compile(pattern, re.DOTALL | re.IGNORECASE)

    if os.path.exists(ham_dir):
        ham_files = os.listdir(ham_dir)
        ham_file_num = len(ham_files)

        labels.append([0] * ham_file_num)
        for file in ham_files:
            full_file_name = r'%s\%s' % (ham_dir, file)
            content = open(full_file_name, 'r', encoding='ISO-8859-1').read()
            words_list.append(
                [word.lower() for word in reg.split(content) if len(word) > 0]
            )

    if os.path.exists(spam_dir):
        spam_files = os.listdir(spam_dir)
        spam_file_num = len(spam_files)

        labels.append([1] * spam_file_num)
        for file in spam_files:
            full_file_name = r'%s\%s' % (spam_dir, file)
            content = open(full_file_name, 'r', encoding='ISO-8859-1').read()
            words_list.append(
                [word.lower() for word in reg.split(content) if len(word) > 0]
            )
    return words_list, np.ravel(labels)


def create_vocabulary_list(word_list):
    vocabulary = set([])
    for words in word_list:
        vocabulary |= set(words)
    return list(vocabulary)


def words2vector_bow(vocabulary, words):
    word_vector = [0] * len(vocabulary)

    for w in words:
        if w in vocabulary:
            word_vector[words.index(w)] += 1
        else:
            print('"%s" is not in the vocabulary' % w)
    return word_vector


def train_NB(train_dataset, labels):
    m, n = train_dataset.shape
    unique_clz = np.unique(labels)
    uniqure_clz_count = len(unique_clz)

    # Laplace Smoothing
    word_in_clz_prob = np.ones((uniqure_clz_count, n))
    clz_prob = np.ones((uniqure_clz_count, 1))

    for i in range(uniqure_clz_count):
        specific_clz_data = train_dataset[labels == unique_clz[i]]
        clz_prob[i, 0] = len(specific_clz_data) / float(n)

        for word_vect in specific_clz_data:
            word_in_clz_prob[i, :] += word_vect

        # prevent underflow
        word_in_clz_prob[i, :] = np.log(
            word_in_clz_prob[i, :] / (np.sum(specific_clz_data) + n)
        )
    # prevent underflow
    clz_prob = np.log(clz_prob)

    classify_NB.pwc = word_in_clz_prob
    classify_NB.pc = clz_prob
    classify_NB.plabels = unique_clz
    return word_in_clz_prob, clz_prob, unique_clz


def classify_NB(test_dataset):
    m = test_dataset.shape[0]
    result = []

    for i in test_dataset:
        pcw = np.sum(classify_NB.pwc * i, axis=1) + classify_NB.pc.ravel()
        print('pcw', pcw)
        idx = np.argmax(pcw)
        result.append(classify_NB.plabels[idx])
    return result


if __name__ == '__main__':
    ham_dir = r'E:\PycharmProjects\ml_impl\examples\bayes\data\email\ham'
    spam_dir = r'E:\PycharmProjects\ml_impl\examples\bayes\data\email\spam'

    word_list, labels = parse_txt2token(ham_dir, spam_dir)

    vocabulary = create_vocabulary_list(word_list)
    print('vocabulary len: ', len(vocabulary))

    train_word_vectors = []
    for doc in word_list:
        train_word_vectors.append(words2vector_bow(vocabulary, doc))

    train_word_vectors = np.array(train_word_vectors)
    test_word_vectors = []
    test_real_result = []
    for i in range(9):
        idx = int(np.random.uniform(0, train_word_vectors.shape[0] - 1))
        test_word_vectors.append(train_word_vectors[idx, :])
        test_real_result.append(labels[idx])
        np.delete(train_word_vectors, idx, axis=1)
        np.delete(labels, idx)

    pwc, pc, plabels = train_NB(train_word_vectors, labels)

    document = ['home', 'based', 'business', 'opportunity',
        'knocking', 'your', 'door', 'don', 'rude', 'and', 'let', 'this', 'chance',
        'you', 'can', 'earn', 'great', 'income', 'and', 'find', 'your',
        'financial', 'life', 'transformed', 'learn', 'more', 'here', 'your',
        'success', 'work', 'from', 'home', 'finder', 'experts']
    word_vector = words2vector_bow(vocabulary, document)
    result = classify_NB(np.array([word_vector]))

    test_result = classify_NB(np.array(test_word_vectors))
    errot_rate = np.sum(np.array(test_result) != np.array(test_real_result)) / float(len(test_word_vectors))
    print('error rate: ', errot_rate)