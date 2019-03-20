import sys
sys.path.append('..')

import pytest
import numpy as np
import entropy_of_behavior as eob  
from predictors.continuous import taylor
class TestEntropyOfBehavior(object):
    """tests for entropy of behavior functions"""

    # entropy_of_behavior.py tests
    def test_update_command_list(self):
        def f(x):
            return x
        
        #not using a list originally (only good for 1d)
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 1)
        entropyObject.update_command_list(np.array([0]))
        assert entropyObject.u_true.tolist() == np.array(
            [[0]]).tolist()

        entropyObject.update_command_list(np.array([1]))
        assert entropyObject.u_true.tolist() == np.array(
            [[0],[1]]).tolist()
        entropyObject.update_command_list(np.array([2]))
        assert entropyObject.u_true.tolist() == np.array([[0], [1], [2]]).tolist()

        entropyObject.update_command_list(np.array([3]))
        assert entropyObject.u_true.tolist() == np.array(
            [[1], [2], [3]]).tolist()

        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 2)
        entropyObject.update_command_list(np.array([0, 0]))
        entropyObject.update_command_list(np.array([1, 1]))
        entropyObject.update_command_list(np.array([2, 2]))
        assert entropyObject.u_true.tolist() == np.array([[0, 0], [1, 1], [2, 2]]).tolist()

        entropyObject.update_command_list(np.array([3, 3]))
        assert entropyObject.u_true.tolist() == np.array([[1, 1], [2, 2], [3, 3]]).tolist()

    def test_check_command(self):

        def f(x):
            return x
        
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 1)
        assert entropyObject.check_command(np.array([0]))

        with pytest.raises(ValueError): 
            entropyObject.check_command([0])

        with pytest.raises(ValueError): 
            entropyObject.check_command(np.array([[0]]))

        with pytest.raises(ValueError):
            entropyObject.check_command(0)
    
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 3)
        assert entropyObject.check_command(np.array([0, 2, 3]))
        with pytest.raises(ValueError):
            entropyObject.check_command(np.array([[0,2,3]]))

        with pytest.raises(ValueError): # "dimension of u does not match N"):
            entropyObject.check_command(np.array([0]))

        with pytest.raises(ValueError): # "dimension of u does not match N"):
            entropyObject.check_command(np.array([0,1,2,3]))
    
    def test_predict_u(self):
        def f1(x):
            return np.array([x[-1,0] + 1])  #just return the last one + 1

        # not enough samples to make prediction case
        entropyObject = eob.EntropyOfBehavior(f1, 3, 3, 1)
        entropyObject.u_true = np.array([[0], [1]])
        entropyObject.predict_u()
        assert entropyObject.u_predicted == []

        entropyObject.u_true = np.array([[0], [1], [2]])
        entropyObject.predict_u()
        assert entropyObject.u_predicted.tolist() == np.array([[3]]).tolist()
        
        #function returns wrong array type
        def f2(x):
            return [x[-1][0] + 1]  # just return the last one + 1
        entropyObject = eob.EntropyOfBehavior(f2, 3, 3, 1)
        entropyObject.u_true = np.array([[0], [1], [2]])
        with pytest.raises(ValueError):
            entropyObject.predict_u()

class TestEntropyOfBehaviorContinuous(object):
    def test_estimate_entropy(self):
        #this is for taylor series so certain parameters are certain
        w_p =  3
        w_e = 10       
        alpha = 0.5    # typically calculated on high performing data
        N = 1          # only thing implemented right now...
        def predictor(x):
            return taylor.predict_u(x)   # this is tested in test_entropy.py

        entropyObject = eob.ContinuousEntropyOfBehavior(
            predictor, w_p, w_e, N, alpha)

        num_test_commands = 13
        commands = np.array([[float(i)/float(num_test_commands)] for i in range(num_test_commands)])
        
        entropyObject.u_true = commands[0:N-1]
        entropyObject.u_predicted = commands[0:N-1]
        assert not entropyObject.estimate_entropy()
        assert not entropyObject.entropy

        #shoud be perfect prediction
        entropyObject.u_true = commands[0:w_e]
        entropyObject.u_predicted = commands[0:w_e]

        actual_entropy = 1e-5*np.log(1e-5)*9  # 8 empty bins
        actual_entropy += 1.0*np.log(1.0)*1
        bins = 10
        actual_entropy = -actual_entropy/np.log(bins)
        assert entropyObject.estimate_entropy() 
        assert entropyObject.entropy == pytest.approx(actual_entropy)
        
        

    def test_update(self):
        assert 0 == 0
