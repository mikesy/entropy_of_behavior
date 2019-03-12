import sys
sys.path.append('..')

import unittest
import entropy_of_behavior as eob  


class EntropyOfBehaviorTestCase(unittest.TestCase):
    """tests for entropy of behavior functions"""

    # entropy_of_behavior.py tests
    def test_update_command_list(self):
        def f(x):
            return x
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 1)
        entropyObject.update_command_list([0])
        entropyObject.update_command_list([1])
        entropyObject.update_command_list([2])
        self.assertEqual(entropyObject.u_for_prediction, [[0],[1],[2]])

        entropyObject.update_command_list([3])
        self.assertEqual(entropyObject.u_for_prediction, [[1], [2], [3]])

        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 2)
        entropyObject.update_command_list([0, 0])
        entropyObject.update_command_list([1, 1])
        entropyObject.update_command_list([2, 2])
        self.assertEqual(entropyObject.u_for_prediction, [[0, 0], [1, 1], [2, 2]])

        entropyObject.update_command_list([3, 3])
        self.assertEqual(entropyObject.u_for_prediction, [[1, 1], [2, 2], [3, 3]])

    def test_check_command(self):
        def f(x):
            return x
        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 1)
        self.assertTrue(entropyObject.check_command([0]))

        entropyObject = eob.EntropyOfBehavior(f, 3, 3, 3)
        self.assertTrue(entropyObject.check_command([0,2,3]))

        with self.assertRaisesRegex(ValueError, "dimension of u does not match N"):
            entropyObject.check_command([0])

        with self.assertRaisesRegex(ValueError, "dimension of u does not match N"):
            entropyObject.check_command([0,1,2,3])
