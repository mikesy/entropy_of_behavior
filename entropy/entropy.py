# shared functions for entropy calculation amongst all models and methods
import numpy as np

def calc_alpha(u_errors, search_tol=1):
    """
    description:
        calculates alpha such that 90% of values fit within the first bucket
        for future calculations
    inputs:
        
    outputs:
        alpha- bucket size for future entropy calculation
    """
    # pass through a single list of values for clean alpha calculation
    
    percent_alpha = 0.9
    n_percent = int(percent_alpha*len(u_errors))     #number for n percent of errors
    max_err = max(u_errors)
    min_err = min(u_errors)
    if max_err > min_err:
        alpha = 0.9*max_err
    else:
        alpha = 0.9*abs(min_err)

    x_alpha_range = []
    n_err_old1 = len(u_errors) + 1 #set justnoutside range to prevent trigger
    n_err_old2 = n_err_old1
    step= 0.1
    #SEARCH_TOL = 4
    x_alpha_range = [n for n in u_errors if abs(n) <= alpha]
    n_err = n_percent -len(x_alpha_range)
    while abs(n_err) > search_tol:
        if n_err == n_err_old2 and n_err != n_err_old1:
            step = step/2
        n_err_old2 = n_err_old1
        n_err_old1 = n_err
        alpha = alpha + step*np.sign(n_err)
        x_alpha_range = [n for n in u_errors if abs(n) <= alpha]
        n_err = n_percent - len(x_alpha_range)

    return alpha

def calc_entropy(u_errors, alpha):
    # can accept a list of lists since data is segmented previously or a single list
    bin_ranges = [-float("inf"), -5*alpha, -2.5*alpha, -alpha, -.5*alpha, .5*alpha, alpha, 2.5*alpha, 5*alpha, float("inf")]
    segment_entropies = []
    for segment_errors in u_errors:
        entropy = 0
        for bin_i in range(0,len(bin_ranges)-1):
            vals_in_range = [n for n in segment_errors if n >= bin_ranges[bin_i]]
            vals_in_range = [n for n in vals_in_range  if n < bin_ranges[bin_i+1]]
            p_bin = float(len(vals_in_range))/float(len(segment_errors))  #probability of bin_i
            if p_bin < 1e-5:
                p_bin = 1e-5
            entropy += p_bin*np.log(p_bin)/np.log(len(bin_ranges)-1)     #uses log rule: log_x (y) = log_a (y) / log_a(x)
        segment_entropies += [-entropy]
    return segment_entropies
