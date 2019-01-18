import numpy as np
from colorama import Fore

def segment_array_using_marker(topic_data, topic_time, marker_times, lag):
    """
    Inputs:
        topic_array = [N x M] array 
            N = number of samples
            M = number of elements in data 
        topic_time = [1 x N] array
        marker_times = [P x 1] array
            P = number of markers 

    Outputs:
        segmented_data = [P x N x lag] array
    """
    if isinstance(topic_data, np.ndarray):
        data_is_list = False
    elif isinstance(topic_data, list):
        data_is_list = True
        topic_data = np.array(topic_data)
    else:
        raise ValueError("unknown data type")

    segmented_data = []
    segmented_time = []
    for marker_time in marker_times:
        marker_pos = get_marker_pos(topic_time, marker_time)
        if marker_pos - lag < -1:
            lag_pos = 0
            print("[WARNING] one time step doesn't have enough time for a lag segment")
        else:
            lag_pos = marker_pos-lag + 1

        if data_is_list:
            segment_data = np.array(topic_data[lag_pos:marker_pos+1])
            segment_time = np.array(topic_time[lag_pos:marker_pos+1])
        else:
            segment_data = topic_data[lag_pos:marker_pos+1]
            segment_time = topic_time[lag_pos:marker_pos+1]
        segmented_data.append(segment_data)
        segmented_time.append(segment_time)

    if data_is_list:
        print("do something to data to return to list")

    segmented_data = np.asarray(segmented_data)
    segmented_time = np.asarray(segmented_time)
    # likely we don't ever need the times but just in case
    return segmented_time, segmented_data


def get_marker_pos(topic_time, marker_time,
                   error_threshold=0.1):
    """
    Inputs:


    Output:
        marker_pos
    """
    marker_pos = -1  # some number that's not possible
    marker_not_found = True
    if topic_time[0] > marker_time:
        raise ValueError(
            "no feasible time bc marker_time is before time starts")
    elif topic_time[-1] < marker_time:
        if marker_time-topic_time[-1] <= error_threshold:
            marker_pos = len(topic_time)
            marker_not_found = False

    while marker_not_found:
        for t_i, t in enumerate(topic_time):
            if t - marker_time > 0.0:  # marker time surpassed
                marker_pos = t_i - 1
                marker_not_found = False
                break

    return marker_pos
