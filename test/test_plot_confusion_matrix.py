# encoding: utf-8
"""
 Author: govind
 Date: 2018/4/20
 Description: 
"""
from plot_tools.metrics import plot_confusion_matrix
import  numpy as np

if __name__ == "__main__":
    A = np.array([[26,  1,  2,  0,  0,  2],
                  [4,  7,  5,  0,  5,  3],
                  [1,  2, 14,  2,  8,  3],
                  [5,  4,  7,  3,  7,  5],
                  [0,  0, 10,  2, 10, 12],
                  [1,  0,  4,  0, 13, 12]])

    x_ticklabels = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']
    y_label = "True Class"
    x_label = "Predicted Class"
    title = "Confusion Matrix of FFT Classifier"
    # plot_confusion_matrix(A, x_label=x_label, y_label=y_label)
