#!/usr/bin/env python
from __future__ import print_function

import argparse, os

import numpy as np

import json
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# Get the data for training
from ml_localization import get_data, models, get_formatters

data_folder = "/data/robin/ml_loc_data"
metadata_fn = os.path.join(data_folder, "20180208-172045_metadata_train_test.json.gz")
metadata_perfmodel_fn = os.path.join(
    data_folder, "metadata_train_test_test_model_alpha_1.0.json"
)


def main():

    # list of available GPUs
    devices = {
        "none": None,
        "main": 0,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
    }

    parser = argparse.ArgumentParser(
        description=(
            "Training of fully conncted newtork for indoor acoustic localization."
        )
    )
    parser.add_argument(
        "config", type=str, help="The config file for the training, model, and data."
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=100,
        help="Number of images in each mini-batch",
    )
    parser.add_argument(
        "--frequency", "-f", type=int, default=-1, help="Frequency of taking a snapshot"
    )
    parser.add_argument(
        "--out", "-o", default="result", help="Directory to output the result"
    )
    parser.add_argument(
        "--gpu",
        default="main",
        choices=devices.keys(),
        help="The GPU to use for the training",
    )
    parser.add_argument(
        "--resume", "-r", default="", help="Resume the training from snapshot"
    )
    parser.add_argument(
        "--noplot",
        dest="plot",
        action="store_false",
        help="Disable PlotReport extension",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    gpu = args.gpu
    epoch = config["training"]["epoch"]
    batchsize = config["training"]["batchsize"]
    out_dir = config["training"]["out"] if "out" in config["training"] else "result"

    print("# Minibatch-size: {}".format(batchsize))
    print("# epoch: {}".format(epoch))
    print("")

    chainer.cuda.get_device_from_id(devices[gpu]).use()

    # Set up a neural network to train
    # Classifier reports mean squared error
    nn = models[config["model"]["name"]](
        *config["model"]["args"], **config["model"]["kwargs"],
    )

    model = L.Classifier(nn, lossfun=F.mean_squared_error)
    # model = L.Classifier(nn, lossfun=F.mean_absolute_error)
    model.compute_accuracy = False

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Helper to load the dataset
    data_formatter, label_formatter, skip = get_formatters(
        **config["data"]["format_kwargs"]
    )

    # Load the dataset
    train, validate, test = get_data(
        config["data"]["file"],
        data_formatter=data_formatter,
        label_formatter=label_formatter,
        skip=skip,
    )

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    validate_iter = chainer.iterators.SerialIterator(
        validate, batchsize, repeat=False, shuffle=False
    )

    # Set up a trainer
    # updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)
    updater = training.StandardUpdater(train_iter, optimizer, device=devices[gpu])
    trainer = training.Trainer(updater, (epoch, "epoch"), out=out_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(validate_iter, model, device=devices[gpu]))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph("main/loss"))

    # Take a snapshot for each specified epoch
    frequency = epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, "epoch"))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ["main/loss", "validation/main/loss"], "epoch", file_name="loss.png"
            )
        )
        trainer.extend(
            extensions.PlotReport(
                ["main/accuracy", "validation/main/accuracy"],
                "epoch",
                file_name="accuracy.png",
            )
        )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "validation/main/loss",
                "main/accuracy",
                "validation/main/accuracy",
                "elapsed_time",
            ]
        )
    )

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # save the trained model
    chainer.serializers.save_npz(config["model"]["file"], nn)

    return nn, train, test


if __name__ == "__main__":
    nn, train, test = main()
