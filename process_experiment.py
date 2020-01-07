import argparse
import os
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal import fftconvolve
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from viewer import BlinkyFile, blinky_non_linearity_inv

from blinky import (
    blinky_signal_aggregation,
    calibration,
    rescale,
    open_recorded_signals,
)
from resampling import (
    compute_audio_power,
    calibrate_and_resample,
)
from utils import snr_db, rmse, decibels, average_absolute_deviation


CM2INCH = 0.39
FOLDER_FIG = Path("figures")


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

    # One of the signals was very long and created very long processing time
    # hackish way of reducing the processing time for that signal
    if str(args.folder).endswith("3x3"):
        experiment_signal = experiment_signal[600:4700, :]

    # now apply the calibration to signal recorded by blinky
    experiment_signal_cal = rescale(experiment_signal, blinky_signal_bounds)
    experiment_signal_inv = blinky_non_linearity_inv(experiment_signal_cal)

    # Now we open the recordings
    audio_fs, audio = open_recorded_signals(args.folder)

    # Calibrate the frame rate and offsets
    # and compute audio power accordingly
    audio_pwr, blinky_cut, b_fps_opt, offset_opt, delays_opt = calibrate_and_resample(
        audio,
        experiment_signal_inv,
        audio_fs,
        blinky_fps_lo_init=29.1,  # it was empirically established that FPS ~ 29.15
        blinky_fps_hi_init=29.2,
        blinky_fps_steps=100,
        n_iter=3,
        cache=True,
        cache_folder=args.folder,
    )

    # Get a raw blinky signal of the same length
    blinky_cut_raw = np.column_stack(
        [
            experiment_signal[delays_opt[n] : delays_opt[n] + blinky_cut.shape[0], n]
            for n in range(experiment_signal.shape[1])
        ]
    )

    # Print the average difference in decibels
    print("Signal length:", blinky_cut.shape[0] / np.mean(b_fps_opt), "seconds")
    print(
        "Average error in decibel domain:",
        average_absolute_deviation(decibels(blinky_cut), decibels(audio_pwr)),
        "decibels",
    )

    # Now we will make the plots, but before that, we create the folder if needed
    if not FOLDER_FIG.exists() or not FOLDER_FIG.is_dir():
        os.mkdir(FOLDER_FIG)

    # Some matplotlib preliminaries for pdf output
    plt.rcParams["pdf.fonttype"] = 42
    # plt.rcParams["font.family"] = "Calibri"
    plt.rcParams["font.family"] = "CMU Sans Serif"

    # Plot a comparison graph
    sns.set_context("paper")
    pal = sns.color_palette("Paired")

    t_inter = np.r_[62.5, 65.5]
    time_vec = np.arange(blinky_cut.shape[0]) / np.mean(b_fps_opt)

    fig, (uncalib, calib) = plt.subplots(1, 2, figsize=(17.6 * CM2INCH, 6 * CM2INCH))
    fig.set_tight_layout({"pad": 0.5})

    for i in [0, 1]:
        uncalib.plot(
            time_vec, blinky_cut_raw[:, i], color=pal[2 * i + 1], label=f"Blinky {i+1}",
        )
    for i in [0, 1]:
        uncalib.add_collection(
            LineCollection(
                [
                    [[time_vec[-n], blinky_signal_bounds[i][m]] for n in [0, 1]]
                    for m in [0, 1]
                ],
                label=f"Calibrated min/max {i+1}",
                color=pal[2 * i],
            )
        )
    uncalib.set_xlabel("Time [ms]")
    uncalib.legend(fontsize="x-small", loc="upper left")
    uncalib.set_title("Raw Blinky Signal")
    uncalib.set_ylabel("Pixel value")
    uncalib.set_xlim(t_inter)
    uncalib.set_ylim([0, np.ceil(np.max(blinky_signal_bounds)) + 10])

    calib.plot(
        time_vec, decibels(blinky_cut[:, 0]), "-", color=pal[1], label="Blinky 1"
    )
    calib.plot(
        time_vec, decibels(blinky_cut[:, 1]), "-", color=pal[3], label="Blinky 2"
    )
    calib.plot(
        time_vec, decibels(audio_pwr[:, 0]), "--", color=pal[0], label="Microphone 1",
    )
    calib.plot(
        time_vec, decibels(audio_pwr[:, 1]), "--", color=pal[2], label="Microphone 2",
    )
    calib.legend(fontsize="x-small")
    calib.set_title("Power After Calibration")
    calib.set_xlabel("Time [s]")
    # Subscript 10 is \u2081\u2080 in unicode, however,
    # my system computer modern font do not support it
    # we use instead small capitals
    calib.set_ylabel("10 log\uf731\uf730(Power)")
    calib.set_xlim(t_inter)

    fig.savefig(FOLDER_FIG / Path("figure_blinky_calibration.pdf"))
    fig.savefig(FOLDER_FIG / Path("figure_blinky_calibration.png", dpi=300))

    plt.show()
