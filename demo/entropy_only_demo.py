#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import sys
sys.path.append('../')
from entropy import entropy


###################################
# 1 dimensional exploration functions
###################################0.001

def normal_distribution_entropy_comparison():
    # create an initial normal distribution
    mu = 0
    sigmas = [0.1, 0.01, 0.2, .4 ]
    samp_num = 1000
    xlim_val = 4.0*sigmas[-1]
    fig = plt.figure(0)
    for s_i, sigma in enumerate(sigmas):
        plt.subplot(1, len(sigmas), s_i+1)
        x = mu + sigma*np.random.randn(samp_num)

        #calculate alpha
        if s_i == 0:
            alpha = entropy.calc_alpha(x)
        entropy_val = entropy.calc_entropy([x], alpha)
        plt.title("H = % 6.4f" % entropy_val[0])
        n,bins,patches = plt.hist(x,int(500*sigma))

        bin_borders = [-5*alpha, -3.5*alpha, -2*alpha, -alpha, 0, alpha, 2*alpha, 3.5*alpha, 5*alpha]
        for border in bin_borders:
            plt.axvline(x = border,color = 'r', linestyle='--')
        plt.xlim(-xlim_val,xlim_val)
        plt.ylim(0,samp_num/15)


# calculate the bin size based on a certain percentage within the central bin

def normal_distribution_entropy_comparison_2d():
    # create an initial normal distribution
    mu = [0,0]
    covs = [[[0.1, 0.0], [0.0, 0.1]], 
            [[0.2, 0.0], [0.0, 0.2]], 
            [[0.1, 0.0], [0.0, 0.4]]]
    samp_num = 1000
    gamma = 0.3
    fig = plt.figure(0)
    for c_i, cov in enumerate(covs):
        plt.subplot(1, len(covs), c_i+1)
        x, y = np.random.multivariate_normal(mu, cov, samp_num).T
        #calculate alpha
        if c_i == 0:
            alpha_x = entropy.calc_alpha(x)
            alpha_y = entropy.calc_alpha(y)
            xlim_val = alpha_x*6
            ylim_val = alpha_y*6
        u = np.vstack((x,y))
        entropy_val = entropy.calc_entropy_Nd([u], [alpha_x,alpha_y])
        print(entropy_val)
        plt.title("H = % 6.4f" % entropy_val[0])
        plt.hist2d(x, y, bins=100, norm=mcolors.PowerNorm(gamma))

        bin_borders_x = [-5*alpha_x, -3.5*alpha_x, -2*alpha_x, 
                       -alpha_x, 0, alpha_x, 2*alpha_x, 3.5*alpha_x, 5*alpha_x]
        bin_borders_y = [-5*alpha_y, -3.5*alpha_y, -2*alpha_y, 
                        -alpha_y, 0, alpha_x, 2*alpha_y, 3.5*alpha_y, 5*alpha_y]
        for border in bin_borders_x:
            plt.axvline(x=border, color='r', linestyle='--')
        for border in bin_borders_y:
            plt.axhline(y=border, color='r', linestyle='--')
        plt.xlim(-xlim_val, xlim_val)
        plt.ylim(-ylim_val, ylim_val)
# 

# create a flatter model


##################################
# 2 dimensional exploration functions
##################################

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="demo some stuff", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--norm_1d', dest="norm_1d", action='store_true', default=False, help="displays 1d entropy normal")
    parser.add_argument('--norm_2d', dest="norm_2d",
                        action='store_true', default=False, help="displays 2d entropy normal")
    parser.add_argument('--all', dest="all",
                        action='store_true', default=False, help="displays all demos, beware of plot mania")

    args = parser.parse_args()

    if args.norm_1d or args.all:
        normal_distribution_entropy_comparison()
    if args.norm_2d:
        normal_distribution_entropy_comparison_2d()
    plt.show()
