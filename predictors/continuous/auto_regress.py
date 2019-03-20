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
        ar_model = VAR(x_train)
    return params




# look at entropic vectors for 2d + in dimensions