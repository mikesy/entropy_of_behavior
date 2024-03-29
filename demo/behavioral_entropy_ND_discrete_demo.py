from matplotlib import colors
import sys
sys.path.append('../')
sys.path.append('../../')

import argparse
from entropy.discrete import discrete_entropy_functions as discrete
import itertools
import matplotlib.pyplot as plt
import numpy as np

figure_i = 0

class CompareDNDEB:

    def __init__(self, save_plots):
        self.figure_i = 0
        self.save_plots = save_plots
        self.classes = ["1", "2", "3"]

        self.create_single_plot_multiple_examples()
        # self.create_single_plot_multiple_examples_off_diag()

    def create_single_plot_multiple_examples(self):

        fig = plt.figure(self.figure_i)
        plt.rc('text', usetex=True)
        self.figure_i += 1
        normalize = False
        if normalize:
            vmax = 1
        else:
            vmax = 99
        norm = colors.Normalize(vmin=0, vmax=vmax)
        plt.subplot(151)
        cm = np.array([[33, 0, 0], [0, 33, 0], [0, 0, 33]])
        self.plot_confusion_matrix_for_subplot(cm,norm, show_y_ticks=True,normalize=normalize)
        plt.ylabel(r'True Command $\textbf{u}$')
        plt.xlabel('(a)')
        
        plt.subplot(152)
        cm = np.array([[31, 1, 1], [1, 31, 1], [1, 1, 31]])
        self.plot_confusion_matrix_for_subplot(cm,norm,normalize=normalize)
        plt.xlabel('(b)')

        plt.subplot(153)
        cm = np.array([[11, 11, 11], [11, 11, 11], [11, 11, 11]])
        self.plot_confusion_matrix_for_subplot(cm,norm,normalize=normalize)
        plt.xlabel('(c)')

        plt.subplot(154)
        cm = np.array([[3, 15, 15], [15, 3, 15], [15, 15, 3]])
        self.plot_confusion_matrix_for_subplot(cm,norm,normalize=normalize)
        plt.xlabel('(d)')

        plt.subplot(155)
        cm = np.array([[80, 0, 0], [1, 8, 1], [2, 2, 5]])
        self.plot_confusion_matrix_for_subplot(cm, norm, normalize=normalize)
        plt.xlabel('(e)')

        # ax.set_xlabel('True Label')
        fig.text(0.525, 0.28, r'Predicted Command $\tilde{\textbf{u}}$', ha='center', va='center')
        if self.save_plots:
            plt.savefig('../plots/comparison_of_5_discrete_entropy_types.eps',
                        format='eps', dpi=1000, bbox_inches='tight')
            plt.savefig('../plots/comparison_of_5_discrete_entropy_types.png',
                        format='png', dpi=1000, bbox_inches='tight')

    def plot_confusion_matrix_for_subplot(self, cm, norm,
                              show_y_ticks=False,
                              normalize=False,
                              cmap=plt.cm.Greens,
                              use_off_diags=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        if use_off_diags:
            H = discrete.calc_entropy_with_off_diags(cm)
            title = "$H_t$ = %0.2f, %0.2f" %(H[0], H[1])
        else:
            H = discrete.calc_entropy(cm)
            title = "$H_t$ = %0.2f" % H

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        im.set_norm(norm)
        plt.title(title)

        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes)#, rotation=45)
        if show_y_ticks:
            plt.yticks(tick_marks, self.classes)
        else:
            plt.yticks(tick_marks, self.classes,visible=False)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 1.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()

    def create_single_plot_multiple_examples_off_diag(self):
        fig = plt.figure(self.figure_i)
        self.figure_i += 1
        normalize = False
        if normalize:
            vmax = 1
        else:
            vmax = 99
        norm = colors.Normalize(vmin=0, vmax=vmax)
        plt.subplot(151)
        cm = np.array([[33, 0, 0], [0, 33, 0], [0, 0, 33]])
        self.plot_confusion_matrix_for_subplot(
            cm, norm, show_y_ticks=True, normalize=normalize, use_off_diags=True)
        plt.ylabel('True Label')
        plt.xlabel('(a)')

        plt.subplot(152)
        cm = np.array([[31, 1, 1], [1, 31, 1], [1, 1, 31]])
        self.plot_confusion_matrix_for_subplot(cm, norm, normalize=normalize, use_off_diags=True)
        plt.xlabel('(b)')

        plt.subplot(153)
        cm = np.array([[11, 11, 11], [11, 11, 11], [11, 11, 11]])
        self.plot_confusion_matrix_for_subplot(cm, norm, normalize=normalize, use_off_diags=True)
        plt.xlabel('(c)')

        plt.subplot(154)
        cm = np.array([[3, 15, 15], [15, 3, 15], [15, 15, 3]])
        self.plot_confusion_matrix_for_subplot(cm, norm, normalize=normalize, use_off_diags=True)
        plt.xlabel('(d)')

        plt.subplot(155)
        cm = np.array([[80, 0, 0], [1, 8, 1], [2, 2, 5]])
        self.plot_confusion_matrix_for_subplot(cm, norm, normalize=normalize, use_off_diags=True)
        plt.xlabel('(e)')

        # ax.set_xlabel('True Label')
        fig.text(0.525, 0.28, 'Predicted label', ha='center', va='center')
        if self.save_plots:
            plt.savefig('../plots/comparison_of_5_discrete_entropy_types.eps',
                        format='eps', dpi=1000, bbox_inches='tight')
            plt.savefig('../plots/comparison_of_5_discrete_entropy_types.png',
                        format='png', dpi=1000, bbox_inches='tight')

if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description="demo some stuff")
    parser.add_argument('--save_plots', dest="save_plots", action='store_true',
                        default=False, help="save plots to ../plots")
    parser.add_argument('--show_plots', dest="show_plots", action='store_true',
                        default=False, help="show plots plt.show()")

    args = parser.parse_args()
    CompareDNDEB(args.save_plots)


    if args.show_plots:
        plt.show()
