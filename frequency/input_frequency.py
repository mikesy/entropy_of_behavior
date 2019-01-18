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


def calculate_fft_frequency_peaks(u, 
                                  t_total = None,
                                  threshold_ratio=0.5,
                                  num_steps = 100):
    """
    Inputs:
        u - [N x M] command sequence
            N = number of timesteps or samples
            M = dimension of control command. (typically 1 to 3D)
        t_total - [N x 1] array
    Outputs:
        peaks = [peaks in Hz] TODO not necessarily Hz if t_total not given

    """

    spectrum = np.fft.fft(u, axis=0)    
    spectrum = np.linalg.norm(spectrum,axis=1)
    if t_total:
        dt = t_total/np.shape(u)[0]
        freq = np.fft.fftfreq(len(spectrum),d=dt)
    else:
        freq = np.fft.fftfreq(len(spectrum))
    threshold = threshold_ratio*np.amax(abs(spectrum))
    mask = np.where(abs(spectrum) > threshold)

    spectrum_at_peaks = spectrum[mask]
    peaks = freq[mask]
    mask = np.where(peaks>0)
    spectrum_at_peaks = spectrum_at_peaks[mask]
    peaks = peaks[mask]

    return peaks, spectrum_at_peaks

def calculate_fft_frequency_spectrum(u,
                                  t_total=None,
                                  threshold_ratio=0.5,
                                  n=100):
    """
    Inputs:
        u - [N x M] command sequence
            N = number of timesteps or samples
            M = dimension of control command. (typically 1 to 3D)
        t_total - [N x 1] array
    Outputs:
        freq
        spectrum

    """
    n = np.shape(u)[0]
    spectrum = np.fft.fft(u, axis=0)
    spectrum = spectrum[0:int(n/2),:]
    spectrum = np.linalg.norm(spectrum, axis=1)
    if t_total:
        dt = t_total/np.shape(u)[0]
        freq = np.fft.fftfreq(n, d=dt)
    else:
        freq = np.fft.fftfreq(n)
    freq = freq[0:int(n/2)]
    return freq, spectrum
