#!/usr/bin/python3


import numpy as np
import unittest

import sys
sys.path.append('../')
from prep import prep

class MatConverterTestCase(unittest.TestCase):
    """tests for entropy functions"""

    def test_segment_array_using_marker(self):

        data = [[0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 10]]

        time = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        lag = 3
        marker_times = [0.42]

        segmented_time, segmented_data = prep.segment_array_using_marker(
            data, time, marker_times, lag)

        self.assertEqual(segmented_data.tolist(), [data[2:5]])
        self.assertEqual(segmented_time.tolist(), [time[2:5]])

        marker_times = [0.42, 0.89]

        segmented_time, segmented_data = prep.segment_array_using_marker(
            data, time, marker_times, lag)

        self.assertEqual(segmented_data.tolist(), [data[2:5], data[6:9]])
        self.assertEqual(segmented_time.tolist(), [time[2:5], time[6:9]])

    def test_get_marker_pos(self):
        times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        marker_time = 0.21
        marker_pos = prep.get_marker_pos(times, marker_time)
        self.assertEqual(marker_pos, 2)

        marker_time = 0.199
        marker_pos = prep.get_marker_pos(times, marker_time)
        self.assertEqual(marker_pos, 1)

        marker_time = 0.001
        marker_pos = prep.get_marker_pos(times, marker_time)
        self.assertEqual(marker_pos, 0)

        marker_time = 0.505
        marker_pos = prep.get_marker_pos(times, marker_time)
        self.assertEqual(marker_pos, 6)

        with self.assertRaisesRegex(ValueError, "no feasible time bc marker_time is before time starts"):
            marker_time = -0.005
            marker_pos = prep.get_marker_pos(times, marker_time)
