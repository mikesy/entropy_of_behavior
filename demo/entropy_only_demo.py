#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from entropy import entropy

###################################
# 1 dimensional exploration
###################################

# create an initial distribution
mu = 0
sigmas = [0.1, 0.001, 0.2, 3 ]

xlim_val = 4.0*sigmas[-1]
for s_i, sigma in enumerate(sigmas):
    plt.subplot(1, len(sigmas), s_i+1)
    x = mu + sigma*np.random.randn(1000)

    #calculate alpha
    if s_i == 0:
        alpha = entropy.calc_alpha(x)
    entropy_val = entropy.calc_entropy([x], alpha)
    plt.title("% 6.4f" % entropy_val[0])
    n,bins,patches = plt.hist(x,50)
    bin_borders = [-5*alpha, -2.5*alpha, -alpha, -.5 *
                  alpha, .5*alpha, alpha, 2.5*alpha, 5*alpha]
    for border in bin_borders:
        plt.axvline(x = border,color = 'r')
    plt.xlim(-xlim_val,xlim_val)


# calculate the bin size based on a certain percentage within the central bin

# 

# create a flatter model



##################################
# 2 dimensional exploration
##################################



plt.show()
