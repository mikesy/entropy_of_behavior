"""
contains all code specific to autoregressive behavioral entropy calculations
"""

import entropy
import numpy as np
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.api import VAR, DynamicVAR

def get_auto_regress_model_params(x_train, lag):
    """
    

    """
    #get shape of x_train
    x_shape = np.shape(x_train)
    if x_shape[0] == 1:
        ar_model = AR(x_train)
        ar_model_fit = ar_model.fit(maxlag=lag,trend='nc')
        params = ar_model_fit.params

    else:  #TODO see if a 1d input to VAR produces the same results as AR
        ar_model = VAR()
    return params



def get_entropy(u, a, alpha):
    """
    inputs:
        u -  the user command
        a -  the coefficients that define the autoregressive model
        alpha - used to calculate entropy
    outputs:
        entropies
    """

    if len(np.shape(u[0])) == 1:
        u_errors = get_command_errors_1d(u, a)
    else:
        raise ValueError("u is not the correct dimension for the given functions")
    # stats
    entropies = entropy.calc_entropy(u_errors,alpha)

    return entropies

def get_command_errors_1d(ucmd,  a):
    #
    #u is control
    # a are ar param
    all_seg_errors = []
    lag = len(a)
    for segment_i in range(0,len(ucmd)):        # cycle through segments split by L2
        u_seg = ucmd[segment_i]                 # segments should be split by the L2 presses
        u_errors = []
        for u_i in range(0,len(u_seg)):
            u_snip = u_seg[u_i:lag+u_i+1]
            if (lag+1+u_i) <= len(u_seg):
                ar_pred = 0
                for s_i in range(lag):          # snippet for prediction
                    ar_pred +=  a[s_i]*u_snip[s_i] 
                pred_error = u_snip[lag] - ar_pred
                u_errors += [pred_error]
        all_seg_errors.append(u_errors)
    return all_seg_errors


# look at entropic vectors for 2d + in dimensions