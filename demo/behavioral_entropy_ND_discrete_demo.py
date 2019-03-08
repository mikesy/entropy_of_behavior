from matplotlib import colors
import sys
sys.path.append('../')
sys.path.append('../../')

import argparse
from entropy import discrete
import itertools
import matplotlib.pyplot as plt
import numpy as np
from my_python_libs.plotting_stats import plot_functions as pf

figure_i = 0

class CompareBEND:

    def __init__(self, save_plots):
        self.figure_i = 0
        self.save_plots = save_plots
        self.classes = ["1", "2", "3"]

        self.create_single_plot_multiple_examples()


    def create_single_plot_multiple_examples(self):
        # fig = plt.figure(self.figure_i)

        fig = plt.figure(self.figure_i)
        self.figure_i += 1
        # ax = fig.add_subplot(111)    # The big subplot
        normalize = False
        if normalize:
            vmax = 1
        else:
            vmax = 99
        norm = colors.Normalize(vmin=0, vmax=vmax)
        plt.subplot(151)
        cm = np.array([[33, 0, 0], [0, 33, 0], [0, 0, 33]])
        self.plot_confusion_matrix_for_subplot(cm,norm, show_y_ticks=True,normalize=normalize)
        plt.ylabel('True Label')
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
        fig.text(0.525, 0.28, 'Predicted label', ha='center', va='center')
        if self.save_plots:
            plt.savefig('../plots/comparison_of_5_discrete_entropy_types.eps',
                        format='eps', dpi=1000, bbox_inches='tight')
            plt.savefig('../plots/comparison_of_5_discrete_entropy_types.png',
                        format='png', dpi=1000, bbox_inches='tight')
        # plt.colorbar(orientation='horizontal')
        
                   # plt.ylabel('True label')
                   # plt.xlabel('Predicted label'))

    def plot_confusion_matrix_for_subplot(self, cm, norm,
                              show_y_ticks=False,
                              normalize=False,
                              cmap=plt.cm.Greens):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        H = discrete.calc_entropy(cm)
        title = "$H_t$ = %0.2f" %H

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

        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.tight_layout()

    def plot_entropy_confusion_matrix(self, cm, title_prefix, file_prefix, plot_colorbar=True):
        H = discrete.calc_entropy(cm)
        plt.figure(self.figure_i)
        pf.plot_confusion_matrix(cm, self.classes, plot_colorbar=plot_colorbar)
        title_str = "%s = %f" % (title_prefix, H)
        plt.title(title_str)
        if self.save_plots:
            plt.savefig('../plots/%s.eps' % file_prefix,
                        format='eps', dpi=1000, bbox_inches='tight')

        self.figure_i += 1 

    # def subplot_entropy_confusion_matrix(self, cm,plot_colorbar=False):
    #     H = discrete.calc_entropy(cm)
    #     pf.plot_confusion_matrix(cm, self.classes, plot_colorbar=plot_colorbar)
    #     plt.title("H = %f" %H)


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description="demo some stuff")
    parser.add_argument('--save_plots', dest="save_plots", action='store_true',
                        default=False, help="save plots to ../plots")
    parser.add_argument('--show_plots', dest="show_plots", action='store_true',
                        default=False, help="show plots plt.show()")

    args = parser.parse_args()
    CompareBEND(args.save_plots)


    if args.show_plots:
        plt.show()
