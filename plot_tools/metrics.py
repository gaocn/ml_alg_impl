# encoding: utf-8
"""
 Author: govind
 Date: 2018/4/20
 Description: 
"""
import matplotlib.pylab as plb
import numpy as np


def plot_confusion_matrix(A, x_label=None, y_label=None, x_ticklabels=None,
                     y_ticklabels=None, title=None):
    # 清屏
    plb.clf()
    plb.matshow(A, fignum=False, cmap='Blues', vmin=np.min(A), vmax=np.max(A))

    # 设置坐标轴
    ax = plb.axes()
    ax.set_xticks(range(len(x_ticklabels)))
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticks(range(len(y_ticklabels)))
    ax.set_yticklabels(y_ticklabels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.xaxis.set_ticks_position('bottom')

    plb.colorbar()
    plb.grid(False)
    plb.show()



