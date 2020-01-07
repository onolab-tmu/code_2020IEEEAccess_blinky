import json
from pathlib import Path
import numpy as np

from utils import decibels, conv_euclidean_distance


FILENAME_CACHE = "resampling_params.json"


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


def calibrate_and_resample(
    audio,
    blinky_pwr,
    audio_fs,
    blinky_fps_lo_init,
    blinky_fps_hi_init,
    blinky_fps_steps,
    n_iter=2,
    cache=True,
    cache_folder=".",
    cache_filename=None,
):
    """
    Finds the best frame rate and offset to compute the power from the audio
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
    n_iter: int
        Number of times to iterate search
    cache: bool
        If true cache the results in a local json file
    """

    n_channels = audio.shape[1]
    assert (
        n_channels == blinky_pwr.shape[1]
    ), "Audio and blinky signals should have the same number of channels"

    if cache_filename is None:
        cache_path = cache_folder / Path(FILENAME_CACHE)
    else:
        cache_path = cache_folder / Path(cache_filename)

    if cache and cache_path.exists():

        with open(cache_path, "r") as f:
            data = json.load(f)

        b_fps_opt = data["fps"]
        offset_opt = data["offset"]
        delays_opt = data["delays"]

    else:

        # match blinky and audio rate
        fps_0 = (blinky_fps_lo_init + blinky_fps_hi_init) / 2  # start at middle
        b_fps_opt = [fps_0 for c in range(n_channels)]
        offset_opt = [0 for c in range(n_channels)]
        delays_opt = [0 for c in range(n_channels)]

        cost = np.zeros((n_channels, blinky_fps_steps))

        for epoch in range(n_iter):

            for n in range(n_channels):

                if n > 0:
                    # make range smaller
                    new_width = (4 / blinky_fps_steps) * (
                        blinky_fps_hi_init - blinky_fps_lo_init
                    )
                    blinky_fps_lo_init = b_fps_opt[n] - new_width / 2
                    blinky_fps_hi_init = b_fps_opt[n] + new_width / 2

                b_fps_range = np.linspace(
                    blinky_fps_lo_init, blinky_fps_hi_init, blinky_fps_steps,
                )

                for i, b_fps in enumerate(b_fps_range):

                    a = compute_audio_power(
                        audio[:, n], audio_fs, b_fps, offset=offset_opt[n]
                    )
                    cost[n, i], delay = conv_euclidean_distance(
                        decibels(blinky_pwr[:, n]), decibels(a)
                    )

                i_opt = np.argmin(cost[n])
                b_fps_opt[n] = float(b_fps_range[i_opt])

            print(f"fps opt: {b_fps_opt}")

            # Compute best offset now
            opt_block_size = np.max([int(audio_fs / v) for v in b_fps_opt])
            print(f"best block {opt_block_size}")
            offset_cost = np.zeros((2, opt_block_size))

            for n in range(n_channels):
                delays = []
                for offset in range(opt_block_size):

                    a = compute_audio_power(
                        audio[:, n], audio_fs, b_fps_opt[n], offset=offset
                    )
                    offset_cost[n, offset], delay = conv_euclidean_distance(
                        decibels(blinky_pwr[:, n]), decibels(a)
                    )
                    delays.append(delay)

                    if offset % 20 == 0:
                        print(f"offset {offset} done")

                i_opt = np.argmin(offset_cost[n])
                delays_opt[n] = int(delays[i_opt])
                offset_opt[n] = int(i_opt)

            print(f"offset opt: {offset_opt}")

        if cache:
            # Save results for later use
            with open(cache_path, "w") as f:
                json.dump(
                    {"fps": b_fps_opt, "offset": offset_opt, "delays": delays_opt}, f
                )

    # Now we compute the best signal and cut to the same length as blinky
    audio_pwr = []
    blinky_cut = []

    for n in [0, 1]:
        audio_pwr.append(
            compute_audio_power(
                audio[:, n], audio_fs, b_fps_opt[n], offset=offset_opt[n]
            )
        )
        blinky_cut.append(blinky_pwr[delays_opt[n] :, n])

    m_len = np.min([len(a) for a in audio_pwr])
    audio_pwr = np.column_stack([a[:m_len] for a in audio_pwr])
    blinky_cut = np.column_stack([a[:m_len] for a in blinky_cut])

    return audio_pwr, blinky_cut, b_fps_opt, offset_opt, delays_opt


