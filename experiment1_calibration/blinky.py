import glob

import numpy as np
from scipy.io import wavfile


def blinky_signal_aggregation(data):
    """
    Aggregate the information from multiple pixels
    """
    return data.astype(np.float).mean(axis=(-2, -1))


def calibration(data):
    """
    Estimate the minimum and maximum of a Blinky signal based on calibration data
    """

    # Find mid point
    mid = (data.min() + data.max()) / 2.0

    # Use the median of upper and lower data as estimator
    lo = np.min(data[data < mid])
    hi = np.median(data[data > mid])

    return [lo, hi]


def rescale(data, bounds):
    """
    Rescale the blinky signals according to the calibration bounds
    """

    output = np.zeros_like(data)

    for c, bnds in enumerate(bounds):
        output[:, c] = (data[:, c] - bnds[0]) / (bnds[1] - bnds[0])

    return output


def open_recorded_signals(path, channel=0):

    audio = []
    fs = None

    for n in [1, 2]:

        filename = glob.glob(str(path / f"rmic{n}*"))[0]
        fs_, audio_ = wavfile.read(filename)

        if fs is None:
            fs = fs_
        else:
            if fs_ != fs:
                raise ValueError("Not all recordings of same sampling frequency")

        assert audio_.dtype == np.int16
        audio_ = audio_ / 2 ** 15

        audio.append(audio_[:, channel])

    m_len = np.min([a.shape[0] for a in audio])
    audio = np.column_stack([a[:m_len] for a in audio])

    return fs, audio[fs // 2 :, :]


