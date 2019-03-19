import sys
sys.path.append('..')

import pytest
import numpy as np
import entropy_of_behavior as eob  
from entropy import taylor
class TestEntropyOfBehavior(object):
    """tests for entropy of behavior functions"""

    # entropy_of_behavior.py tests
    def test_update_command_list(self):
        def f(x):
            return x
        
        #not using a list originally (only good for 1d)
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 1)
        entropyObject.update_command_list(0)
        entropyObject.update_command_list(1)
        entropyObject.update_command_list(2)
        assert entropyObject.u_true == [0, 1, 2]

        #using a list but still 1d (more general use case)
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 1)
        entropyObject.update_command_list([0])
        entropyObject.update_command_list([1])
        entropyObject.update_command_list([2])
        assert entropyObject.u_true == [[0],[1],[2]]

        entropyObject.update_command_list([3])
        assert entropyObject.u_true == [[1], [2], [3]]

        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 2)
        entropyObject.update_command_list([0, 0])
        entropyObject.update_command_list([1, 1])
        entropyObject.update_command_list([2, 2])
        assert entropyObject.u_true == [[0, 0], [1, 1], [2, 2]]

        entropyObject.update_command_list([3, 3])
        assert entropyObject.u_true == [[1, 1], [2, 2], [3, 3]]

    def test_check_command(self):
        def f(x):
            return x
        
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 1)
        assert entropyObject.check_command([0])
        assert entropyObject.check_command(0)

        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 3)
        assert entropyObject.check_command([0,2,3])

        with pytest.raises(ValueError): # "dimension of u does not match N"):
            entropyObject.check_command([0])

        with pytest.raises(ValueError): # "dimension of u does not match N"):
            entropyObject.check_command([0,1,2,3])
    
    def test_predict_u(self):
        def f1(x):
            return x[-1][0] + 1  #just return the last one + 1

        entropyObject = eob.EntropyOfBehavior(f1, 3, 3, 1)

        entropyObject.u_true = [0, 1]
        entropyObject.predict_u()
        assert entropyObject.u_predicted == []

        def f2(x):
            return [x[-1][0] + 1]  # just return the last one + 1

        entropyObject = eob.EntropyOfBehavior(f2, 3, 3, 1)
        #same test using more general use case
        entropyObject.u_true = [[0],[1]]
        entropyObject.predict_u()
        assert entropyObject.u_predicted == []

        entropyObject.u_true = [[0], [1], [2]]
        entropyObject.predict_u()
        assert entropyObject.u_predicted == [[3]]

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
        commands = np.array([float(i)/float(num_test_commands) for i in range(num_test_commands)])

        entropyObject.u_true = commands[0:9]

        # for u in commands:
        #     entropy
        

    def test_update(self):
        assert 0 == 0
