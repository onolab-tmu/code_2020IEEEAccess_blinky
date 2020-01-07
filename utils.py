import numpy as np
from scipy.signal import fftconvolve


def decibels(x):
    return 10.0 * np.log10(x)


def average_absolute_deviation(signal, reference):
    return np.mean(np.abs(signal - reference))


def mse(signal, reference):
    return np.mean(np.abs(signal - reference) ** 2)


def rmse(signal, reference):
    return np.sqrt(mse(signal, reference))


def snr_db(signal, reference):
    return decibels(np.var(reference) / mse(signal, reference))


def itakura_saito(signal, reference):
    return np.sum(reference / signal - np.log(reference / signal) - 1)


def lin_reg(x, y):

    A = np.column_stack([x, np.ones(x.shape[0])])
    return np.linalg.solve(A.T @ A, A.T @ y)


def conv_euclidean_distance(s1, s2):
    """
    Finds the best offset in terms of Euclidean distance
    """

    if len(s1) < len(s2):
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    # efficiently compute the error
    s2_sum2 = (s2 ** 2).sum()
    s1_sum2 = fftconvolve(s1 ** 2, np.ones(len(s2)), mode="valid")
    s1_conv_s2 = fftconvolve(s1, s2[::-1], mode="valid")

    cost = np.sqrt(s2_sum2 - 2 * s1_conv_s2 + s1_sum2)
    best_offset = np.argmin(cost)

    if swapped:
        return cost[best_offset], -best_offset
    else:
        return cost[best_offset], best_offset
