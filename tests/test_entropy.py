import unittest
import sys
sys.path.append('../')

from entropy import entropy, taylor, auto_regress, nn, auto_agree


class EntropyTestCase(unittest.TestCase):
    """tests for entropy functions"""

    # entropy.py tests
    def test_calc_alpha(self):

        #takes only one step to find alpha that contains the 90% within one bucket
        u_errors = [0, 0, 0, 0, 0, 0, 0.5, -0.4, 0.25, .6]
        search_tol = 1
        correct_result = 0.54  
        self.assertEqual(entropy.calc_alpha(u_errors,search_tol), correct_result)
        

    def test_calc_entropy(self):
        #TODO make some hand calculated tests and ensure entropy function works properly
        self.assertEqual(0,0)


    # taylor tests        
    def test_get_command_errors_1d(self):
        #TODO create tests
        self.assertEqual(0,0)
    

    # auto_regress tests

    # nn tests


    #auto_agree tests
