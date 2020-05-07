import numpy as np
from scipy.integrate import simps


def fft(data, fs, band):
    fft_vals = np.absolute(np.fft.rfft(data))
    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)

    freq_ix = np.where((fft_freq >= band[0]) & (fft_freq <= band[1]))[0]
    res = simps(fft_vals[freq_ix])

    return res
