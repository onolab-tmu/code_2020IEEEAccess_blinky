import argparse
import glob
from pathlib import Path

import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal import fftconvolve
from scipy.io import wavfile
from viewer import BlinkyFile, blinky_non_linearity_inv


def decibels(x):
    return 10.0 * np.log10(x)


def snr_db(signal, reference):
    return 10.0 * np.log10(np.var(reference) / np.var(signal - reference))


def itakura_saito(signal, reference):
    return np.sum(reference / signal - np.log(reference / signal) - 1)


def lin_reg(x, y):

    A = np.column_stack([x, np.ones(x.shape[0])])
    return np.linalg.solve(A.T @ A, A.T @ y)


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
    lo = np.median(data[data < mid])
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


def compute_audio_power(audio, audio_fs, power_fs, offset=0):
    """ Compute the power of audio at a fractional sampling rate """

    assert power_fs < audio_fs

    block_size = audio_fs / power_fs
    block_size_i = int(block_size)  # integer part
    block_size_f = block_size - block_size_i  # fractional part

    out_len = int((audio.shape[0] - block_size_i) / block_size)
    pwr = np.zeros(out_len, dtype=audio.dtype)

    a_n = offset  # audio array index
    sample_error = 0.0
    for p_n in range(out_len):  # power array index

        if sample_error <= 1.0:
            this_blk_len = block_size_i
            sample_error += block_size_f
        else:
            this_blk_len = block_size_i + 1
            sample_error -= 1.0

        pwr[p_n] = (audio[a_n : a_n + this_blk_len] ** 2).mean()
        a_n += this_blk_len

    return pwr


def conv_euclidean_distance(s1, s2):
    """
    Finds the best offset
    """

    if len(s1) < len(s2):
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    n_offsets = len(s1) - len(s2) + 1

    tricked = as_strided(
        s1, shape=(n_offsets, len(s2)), strides=(s1.strides[0], s1.strides[0])
    )

    cost = np.linalg.norm(tricked - s2[None, :], axis=1)

    best_offset = np.argmin(cost)

    if swapped:
        return cost[best_offset], -best_offset
    else:
        return cost[best_offset], best_offset


def conv_euclidean_distance2(s1, s2):
    """
    Finds the best offset
    """

    if len(s1) < len(s2):
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    # efficiently compute the error
    s2_sum2 = (s2 ** 2).sum()
    s1_sum2 = fftconvolve(s1 ** 2, np.ones(len(s2)), mode="valid")
    s1_conv_s2 = fftconvolve(s1, s2, mode="valid")

    cost = np.sqrt(s2_sum2 - 2 * s1_conv_s2 + s1_sum2)
    best_offset = np.argmin(cost)

    if swapped:
        return cost[best_offset], -best_offset
    else:
        return cost[best_offset], best_offset


def match_rate_resample(audio, blinky, audio_fs, blinky_fs):
    """
    This function should match the sampling rate and resample the audio
    power at the same time
    """

    assert blinky_fs < audio_fs

    block_size = audio_fs / blinky_fs
    block_size_i = int(block_size)  # integer part
    block_size_f = block_size - block_size_i  # fractional part

    # remove the mean
    audio = audio - audio.mean(axis=0, keepdims=True)

    a_n = 0
    blk_delays = []
    n_blocks = 200
    block_inc = 50

    while a_n + n_blocks * block_size_i < audio.shape[0]:

        a = audio[a_n : a_n + n_blocks * block_size_i]

        the_block = compute_audio_power(a, audio_fs, blinky_fs)

        # delay = pra.tdoa(the_block, blinky, phat=False)
        c, delay = conv_euclidean_distance(blinky, the_block)
        blk_delays.append(delay)

        # plt.plot(blinky, label="blinky")
        # plt.plot(np.arange(-delay, -delay + the_block.shape[0]), the_block)
        # plt.show(block=True)

        a_n += block_inc * block_size_i

    # time = np.arange(len(blk_delays)) * block_inc * block_size_i
    # a, b = lin_reg(time, blk_delays)

    blk_delays = np.array(blk_delays)
    # deltas = np.diff(blk_delays)
    # p = np.percentile(deltas, [25, 75])
    # center = np.logical_and(deltas >= p[0], deltas <= p[1])
    # mean_delta = np.mean(deltas[center])

    plt.plot(blk_delays)
    plt.show(block=True)

    est_block_size = 1.0
    est_blinky_fs = est_block_size / block_size_i * blinky_fs

    # import pdb; pdb.set_trace()

    pwr = compute_audio_power(audio, audio_fs, est_blinky_fs)

    return est_blinky_fs, pwr


def match_rate_resample_search(audio, blinky, video_fps_range):

    # do the power computations with approximate block size
    import librosa
    from numpy.lib.stride_tricks import as_strided

    values = []

    for fps in video_fps_range:
        block_size = int(audio_fs / fps)
        values.append([])
        for offset in range(block_size):
            audio_pwr = compute_audio_power(audio, audio_fs, video_fps)

            c, delay = conv_euclidean_distance(blinky, audio_pwr)

            print(f"fps: {fps} offset: {offset} cost:{c}")

            values[-1].append(
                {
                    "cost": c,
                    "offset": offset,
                    "fps": fps,
                    "delay": delay,
                    "pwr": audio_pwr,
                }
            )

    best = None
    for e in values:
        for g in e:
            if best is None or g["cost"] < best["cost"]:
                best = g

    return best


def match_rate(audio_pwr, blinky_pwr):
    """
    find the best offset

    Parameters
    ----------
    audio_pwr: numpy.ndarray (n_offsets, n_samples, n_blinky)
        The power measured with microphones
    blinky_pwr: numpy.ndarray (n_samples, n_blinky)
        The power measured by blinkies
    """

    for b in range(audio_pwr.shape[-1]):  # blinky index

        a = audio_pwr[:, b]

        xcorr = pra.correlate(a, blinky_pwr[:, b])
        t = np.argmax(np.abs(xcorr)) - blinky_pwr.shape[0] + 1

        a = audio_pwr[:, b]
        plt.figure()
        plt.plot(blinky_pwr[:, b], label="blinky")
        plt.plot(np.arange(-t, -t + a.shape), a, label="recorder")
        plt.legend()
        plt.title(f"matched {b} {t}")


def calibrate_fps_offset(
    audio, blinky_pwr, audio_fs, blinky_fps_lo, blinky_fps_hi, blinky_fps_steps
):
    """
    Finds the best sampling rate and offset to compute the power from the audio
    signal that matches the one recorded by the blinky device using a brute
    force search approach

    Parameters
    ----------
    audio: ndarray (n_samples, n_channels)
        The audio signal
    blinky_pwr: ndarray (n_frames, n_channels)
        The signal recorded by the Blinky
    audio_fs: int
        The sampling rate of the audio signal
    blinky_fps_lo: float
        The low range of the Blinky frame rate estimate
    blinky_fps_hi: float
        The high range of the Blinky frame rate estimate
    blinky_fps_steps: int
        The number of points to divide the interval into for search
    """

    n_channels = audio.shape[1]
    assert (
        n_channels == blinky_pwr.shape[1]
    ), "Audio and blinky signals should have the same number of channels"

    # match blinky and audio rate
    blinky_fps = (blinky_fps_hi + blinky_fps_lo) / 2
    b_fps_range = np.linspace(blinky_fps_lo, blinky_fps_hi, blinky_fps_steps)
    b_fps_opt = [blinky_fps, blinky_fps]
    offset_opt = [0, 0]
    delays_opt = [0, 0]

    cost = np.zeros((2, len(b_fps_range)))

    for epoch in range(2):

        for n in [0, 1]:
            for i, b_fps in enumerate(b_fps_range):

                a = compute_audio_power(
                    audio[:, n], audio_fs, b_fps, offset=offset_opt[n]
                )
                cost[n, i], delay = conv_euclidean_distance(blinky_pwr[:, n], a)

            i_opt = np.argmin(cost[n])
            b_fps_opt[n] = b_fps_range[i_opt]

        print(f"fps opt: {b_fps_opt}")

        # Compute best offset now
        opt_block_size = np.max([int(audio_fs / v) for v in b_fps_opt])
        print(f"best block {opt_block_size}")
        offset_cost = np.zeros((2, opt_block_size))

        for n in [0, 1]:
            delays = []
            for offset in range(opt_block_size):

                a = compute_audio_power(
                    audio[:, n], audio_fs, b_fps_opt[n], offset=offset
                )
                offset_cost[n, offset], delay = conv_euclidean_distance(
                    blinky_pwr[:, n], a
                )
                delays.append(delay)

                if offset % 20 == 0:
                    print(f"offset {offset} done")

            i_opt = np.argmin(offset_cost[n])
            delays_opt[n] = delays[i_opt]
            offset_opt[n] = i_opt

        print(f"offset opt: {offset_opt}")

    # Now we compute the best signal and cut to the same length as blinky
    audio_pwr = []
    blinky_cut = []

    for n in [0, 1]:
        audio_pwr.append(
            compute_audio_power(
                audio[:, n], audio_fs, b_fps_opt[n], offset=offset_opt[n]
            )
        )
        blinky_cut.append(blinky_pwr[delays_opt[n] : delays_opt[n] + len(a), n])

    m_len = np.min([len(a) for a in audio_pwr])
    audio_pwr = np.column_stack([a[:m_len] for a in audio_pwr])
    blinky_cut = np.column_stack([a[:m_len] for a in blinky_cut])

    return audio_pwr, blinky_cut, b_fps_opt, offset_opt, delays_opt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process the data of Blinky experiment"
    )
    parser.add_argument("folder", type=Path, help="Location of experiment data")
    args = parser.parse_args()

    # calibration file
    calibration_file = BlinkyFile.load(args.folder / "calibration.blinky")
    calibration_signal = blinky_signal_aggregation(calibration_file.data)

    blinky_signal_bounds = []
    for i in range(calibration_signal.shape[1]):

        # calibrate
        bnds = calibration(calibration_signal[:, i])
        blinky_signal_bounds.append(bnds)

    # process the data recorded by the blinky
    experiment_file = BlinkyFile.load(args.folder / "experiment.blinky")
    experiment_signal = blinky_signal_aggregation(experiment_file.data)
    video_fps = experiment_file.fps

    # now apply the calibration to signal recorded by blinky
    experiment_signal_cal = rescale(experiment_signal, blinky_signal_bounds)
    experiment_signal_inv = blinky_non_linearity_inv(experiment_signal_cal)

    if str(args.folder).endswith("3x3"):
        experiment_signal_inv = experiment_signal_inv[600:4700, :]

    # Now we open the recordings
    audio_fs, audio = open_recorded_signals(args.folder)

    # match blinky and audio rate
    video_fps = 29.14
    v_fps_range = np.linspace(29.141, 29.145, 100)
    v_fps_opt = [video_fps, video_fps]
    offset_opt = [0, 0]
    delays_opt = [0, 0]

    cost = np.zeros((2, len(v_fps_range)))

    for epoch in range(2):

        for n in [0, 1]:
            for i, v_fps in enumerate(v_fps_range):

                a = compute_audio_power(
                    audio[:, n], audio_fs, v_fps, offset=offset_opt[n]
                )
                cost[n, i], delay = conv_euclidean_distance(
                    experiment_signal_inv[:, n], a
                )

            i_opt = np.argmin(cost[n])
            v_fps_opt[n] = v_fps_range[i_opt]

        print(f"fps opt: {v_fps_opt}")

        # Compute best offset now
        opt_block_size = np.max([int(audio_fs / v) for v in v_fps_opt])
        print(f"best block {opt_block_size}")
        offset_cost = np.zeros((2, opt_block_size))

        for n in [0, 1]:
            delays = []
            for offset in range(opt_block_size):

                a = compute_audio_power(
                    audio[:, n], audio_fs, v_fps_opt[n], offset=offset
                )
                offset_cost[n, offset], delay = conv_euclidean_distance(
                    experiment_signal_inv[:, n], a
                )
                delays.append(delay)

                if offset % 20 == 0:
                    print(f"offset {offset} done")

            i_opt = np.argmin(offset_cost[n])
            delays_opt[n] = delays[i_opt]
            offset_opt[n] = i_opt

        print(f"offset opt: {offset_opt}")

    # Now we compute the best signal and cut to the same length as blinky
    audio_pwr = []
    blinky_cut = []

    for n in [0, 1]:
        audio_pwr.append(
            compute_audio_power(
                audio[:, n], audio_fs, v_fps_opt[n], offset=offset_opt[n]
            )
        )
        blinky_cut.append(
            experiment_signal_inv[delays_opt[n] : delays_opt[n] + len(a), n]
        )

    m_len = np.min([len(a) for a in audio_pwr])
    audio_pwr = np.column_stack([a[:m_len] for a in audio_pwr])
    blinky_cut = np.column_stack([a[:m_len] for a in blinky_cut])

    # plot the figure
    fig, axes = plt.subplots(1, 2)
    for n in [0, 1]:
        axes[n].plot(audio_pwr[:, n], label="audio")
        axes[n].plot(blinky_cut[:, n], label="blinky")
        axes[n].legend()

    # Plot the results
    """
    fig, (uncalib, calib, rec) = plt.subplots(1, 3)
    uncalib.plot(experiment_signal)
    calib.plot(10 * np.log10(experiment_signal_inv))
    rec.plot(10 * np.log10(audio_pwr[0]))
    """

    plt.show()
