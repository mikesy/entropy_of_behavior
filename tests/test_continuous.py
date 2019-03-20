import sys
sys.path.append('../')

from entropy.continuous import continuous_entropy_functions as cef
import numpy as np
import pytest


class TestContinuousEntropy(object):
    """tests for continuous entropy functions"""

    def test_calc_entropy(self):
        #TODO make some hand calculated tests and ensure entropy function works properly
        alpha = 1.0
        u_errors = np.array([-0.5, -0.5, 0.5, 0.5])
        actual_entropy = 1e-5*np.log(1e-5)*8  # 8 empty bins
        actual_entropy += 0.5*np.log(0.5)*2  # 2 bins in center evenly distributed (0.5)
        bins = 10
        actual_entropy = -actual_entropy/np.log(bins)
        assert cef.calc_entropy(u_errors,alpha) == pytest.approx(actual_entropy)
        # assert 0 == 0

    # taylor tests

    
    def test_get_command_errors_1d(self):
        #TODO create tests
        assert 0 == 0
