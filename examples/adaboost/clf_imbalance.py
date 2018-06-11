# encoding: utf-8
"""
 Author: govind
 Date: 2018/6/1
 Description: 
"""
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_roc(pred, y):
    """
     performance metrics: precision, recall, and ROC
     -------------------

     Confusion Matrix
     -------------------
                       Predicted
                Actual |_____T____________F________
                   T   |    TP     |     FN       |
                   F   |    FP     |     TN       |

     Precision = TP / (TP + FP) 预测正例中正确的比例称为精度；
     Recall    = TP / (TP + FN) 正例中被预测正确的比例称为召回率；

    ROC curve and AUC
    --------------------
    ROC（Receiver Operating Characteristic）,其首次被用于二战中构建雷达系统。

    ROC是X轴坐标为FPR(False Positive Rate),Y轴为TPR(True Positive Rate)的图形
    TPR又称为召回率，        定义为：TP / (TP + FN)
    FPR又称为1-specificity，定义为：FP / (FP + TN)

    The ROC curve shows how the two rates change as the threshold changes. The
    leftmost point corresponds to classifying everything as the negative class,
    and the rightmost point corresponds to classifying everything in the positive
    class. The dashed line is the curve you’d get by randomly guessing.

    ROC以概率为阈值来度量模型正确识别正例的比例与模型错误的把负例识别成正例的比例之间的权衡。
    TPR的增加必定以FPR的增加为代价，ROC曲线下方的面积是模型准确率的度量。ROC曲线有助于可以
    作为"cost-versus-benefit decisions"的依据。

    Ideally, the best classifier would be in upper left as much as possible. This
    would mean that you had a high true positive rate for a low false positive rate.

    One metric to compare different ROC curves is the area under the curve (AUC).
    A perfect classifier would have an AUC of 1.0, and random guessing will give
    you a 0.5.

    In order to plot the ROC you need the classifier to give you a numeric score
    of how positive or negative each instance is.

    To build the ROC curve, you first sort the instances by their prediction
    strength. You start with the lowest ranked instance and predict everything
    below this to be in the negative class and everything above this to be the
    positive class. This corresponds to the point 1.0,1.0. You move to the next
    item in the list, and if that is the positive class, you move the true positive
    rate, but if that instance is in the negative class, you change the true negative
    rate.

    绘制ROC要求模型必须能返回监测元组的类预测概率，根据概率对元组排序和定秩，并使正概率较大
    的在顶部，负概率较大的在底部进行画图。

    fpr,tpr,thresholds = sklearn.metrics.roc_curve(y_true,y_score,pos_label=None)

    Score：表示每个测试样本属于正样本的概率；
    thresholds: 分类器的一个重要功能"概率输出"，即表示分类器认为某个样本具有多大的概率属于正样本（或负样本）；
    pos_label: 指定正比例标签

    从高到低，依次将"score"值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们
    认为它为正样本，否则为负样本。每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一
    点。当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。将这些(FPR,TPR)对连接起
    来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。


    【AUC的计算方法】
       1. 样本量少的情况下，得到ROC曲线是阶梯型，计算AUC直接求和阶梯下的面积之和。
         先排序score(score越大，正样本概率越大)，然后按照y_step扫描计算矩阵高度之和乘以矩阵宽x_step即可。
         缺点是：当多个score相同时，得到是一个斜着向上的梯形，这个计算起来比较麻烦！

       2. AUC与Wilcoxon-Mann-Witney Test是等价的
         Wilcoxon-Mann-Witney Test：测试任意给定一个正类和负类样本，正类样本的score有多大概率大于负类样本的score。
         根据大数定理，在样本量越多的情况下，通过频率就可以逼近概率。

          M为：正类样本的数目
          N为：负类样本的数目
         计算方法：统计所有M*N个正负样本对中，有多少个组合的正样本的score大于负样本的score，当二元组中正负样本的score
         相等时，按照0.5计算；然后除以M*N。时间复杂度为：O((M+N)^2)

       3. 第二种方法的改进，复杂度降低了，首先对score从大到小排序，然后令最大score对应的样本的rank为n，第二大的score样本
         的rank为n-1，以此类推。最后把所有正类样本的rank相加，减去正类样本的score最小的M个的rank之和就得到"所有样本中有
         多少对正类样本的score大于负类有样本的score"，再除以M*N就得到AUC的值
                    AUC = [（所有正类rank之和） - M(M+1)/2]  / M*N

        PS：
        1). score相等的样本的rank应相同,具体操作是把所有这些score相等的样本 的rank取平均。
        2). 为了求的组合中正样本的score值大于负样本，如果所有的正样本score值都是大于负样本的，那么第一位与任意的进行组合
            score值都要大，我们取它的rank值为n，但是n-1中有M-1是正样例和正样例的组合这种是不在统计范围内的，所以要减掉，
            那么同理排在第二位的n-1，会有M-1个是不满足的，依次类推，故得到后面的公式M*(M+1)/2，我们可以验证在正样本score
            都大于负样本的假设下，AUC的值为1。

    """
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)

    # to calculate AUC
    y_sum = 0.
    cursor_pos = (1.0, 1.0)
    # this give number of steps in y direction
    num_positive_clz = np.sum(y == 1.0)

    # x,y axes's range is [0, 1]
    x_step = 1.0 / float(len(y) - num_positive_clz)
    y_step = 1.0 / float(num_positive_clz)

    # from smallest to largest, draw from (1,1) to (0,0)
    sorted_indices = np.argsort(pred, axis=0).T[0]

    # loop through all values, drawing a line segment at each point
    for idx in list(sorted_indices):
        if y[idx] == 1.0:
            # take a step down in y direction by decreasing TPR
            delX = 0
            delY = y_step
        else:
            # take a step backward in x direction
            delX = x_step
            delY = 0
            y_sum += cursor_pos[1]

        # draw line from cursor_pos to (cursor_pos[0] - delX, cursor_pos[1] - devY)
        ax.plot([cursor_pos[0], cursor_pos[0] - delX],
                [cursor_pos[1], cursor_pos[1] - delY], c='b')
        cursor_pos = (cursor_pos[0] - delX, cursor_pos[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    AUC = y_sum * x_step

    ax.set_xlabel('False Positive rate')
    ax.set_ylabel('True Positive rate')
    ax.axis([0, 1, 0, 1])
    plt.title('ROC Curve for horse calic detection system(AUC=%.2f)' % AUC)
    plt.show()
    print('Area Under the ROC Curve(AUC) is : ', y_sum * x_step)


if __name__ == '__main__':
    y = np.array([1, 1, -1, -1])
    pred = np.array([0.1, 0.4, 0.35, 0.8])
    y = y.reshape((-1, 1))
    pred = pred.reshape((-1, 1))

    from adaboost_horse_colic import load_data, adaboost
    X, y, testX, testY = load_data()
    classifiers, pred_scores = adaboost(X, y, 10)

    plot_roc(pred_scores, y)