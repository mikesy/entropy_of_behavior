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

def get_errors(u_true, u_pred):
    """
    must be np arrays
    """
    u_errors = u_true-u_pred
    return u_errors

def calc_entropy(u_errors, alpha):
    """
    only 1d right now, TODO make general
    """
    # can accept a list of lists since data is segmented previously or a single list
    bin_ranges = [-float("inf"), -5*alpha, -3.5*alpha, -2*alpha, -alpha, 
                  0, alpha, 2*alpha, 3.5*alpha, 5*alpha, float("inf")]
    entropy = 0
    for bin_i in range(0,len(bin_ranges)-1):
        vals_in_range = [n for n in u_errors if n >= bin_ranges[bin_i]]
        vals_in_range = [n for n in vals_in_range  if n < bin_ranges[bin_i+1]]
        p_bin = float(len(vals_in_range))/float(len(u_errors))  #probability of bin_i
        if p_bin < 1e-5:
            p_bin = 1e-5
        entropy += p_bin*np.log(p_bin)/np.log(len(bin_ranges)-1)     #uses log rule: log_x (y) = log_a (y) / log_a(x)
    return -entropy

def calc_entropy_Nd(u_errors, alphas):
    # can accept a list of lists since data is segmented previously or a single list
    bin_ranges = []
    dim = len(alphas)
    for alpha in alphas:
        bin_ranges.append([-float("inf"), -5*alpha, -3.5*alpha, -2*alpha, -alpha,
                           0, alpha, 2*alpha, 3.5*alpha, 5*alpha, float("inf")])

    all_bin_combos = list(get_nbin_lims(0, len(bin_ranges[0])-1, dim))
    num_bins = len(all_bin_combos)
    segment_entropies = []
    for segment_errors in u_errors:
        num_points = len(segment_errors[0,:])
        entropy = 0
        for bin_combo in all_bin_combos:
            point_count = 0
            for u_i in range(num_points):
                in_bin = True
                for dim_i in range(dim):
                    u = segment_errors[dim_i,u_i]
                    if u <= bin_ranges[dim_i][bin_combo[dim_i]] or u > bin_ranges[dim_i][bin_combo[dim_i]+1]:
                        in_bin = False
                        break
                if in_bin:
                    point_count += 1
                
            p_bin = float(point_count) /float(num_points)  # probability of bin_
            if p_bin < 1e-10:
                p_bin = 1e-10

            entropy += p_bin*np.log(p_bin)/np.log(num_bins) # uses log rule: log_x (y) = log_a (y) / log_a(x)
        segment_entropies += [-entropy]
    return segment_entropies

def get_nbin_lims(start,stop,dims):
    if not dims:
        yield()
        return
    for outer in get_nbin_lims(start,stop,dims-1):
        for inner in range(start,stop):
            yield outer + (inner,)