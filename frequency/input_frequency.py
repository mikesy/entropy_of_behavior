import numpy as np


def calculate_non_zero_frequency(u, t,
                                 nonzero_threshold=1e-3):
    """ 
    Inputs:
        u - [N x M] command sequence
            N = number of timesteps or samples
            M = dimension of control command. (typically 1 to 3D)
        t - [N x 1] array
    Outputs:
        f = frequency of the segment

    """

    sample_time = t[-1] - t[0]
    N = len(t)
    publish_rate = N/sample_time

    nonzero_count = 0
    for u_i in u:
        if np.linalg.norm(u_i) < nonzero_threshold:
            nonzero_count += 1
    
    ratio_nonzero = nonzero_count/N

    f = ratio_nonzero*publish_rate

    return f

def calculate_fft_frequency_peaks(u,t, 
    threshold_ratio = 0.5):
    """
    Inputs:
        u - [N x M] command sequence
            N = number of timesteps or samples
            M = dimension of control command. (typically 1 to 3D)
        t - [N x 1] array
    Outputs:
        f = frequency of the segment

    """

    spectrum = np.fft.fft(u, axis=0)    #n dimensional fft TODO check and write tests
    freq = np.fft.fftfreq(len(spectrum))

    threshold = threshold_ratio*max(abs(spectrum))
    mask = abs(spectrum) > threshold
    peaks = freq[mask]
    return peaks
