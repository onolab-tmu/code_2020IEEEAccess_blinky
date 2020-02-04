Blinky Experiment 2: Localization
=================================

In this experiment, we do sound source localiation with 101 actual Blinkies.

Install
-------

We rely on [Anaconda](https://www.anaconda.com/distribution/) to create the environment.

    conda env create -f environment.yml


Reproduce the experiment
------------------------

The results from the paper can be reproduced by running the following script.

    ./make_all.sh

This script can be used as a reference for all the steps needed from downloading
the data to training the networking and running the test.

    #!/bin/bash

    echo $0
    CODE_DIR=$(dirname $0)
    EXP_DIR="data"

    # Get the data
    python ${CODE_DIR}/get_data.py ${EXP_DIR}/protocol.json

    # This extracts the blinky tracks from the video, as
    # well as the audio source locations when the RC car is used
    python ${CODE_DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v noise --track
    python ${CODE_DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v speech --track
    python ${CODE_DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_1
    python ${CODE_DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_2
    python ${CODE_DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_3
    python ${CODE_DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_4
    python ${CODE_DIR}/experiment_processing.py ${EXP_DIR}/protocol.json -v hori_5

    # Prepare the data extracted from the videos to be fed to the DNN
    python ${CODE_DIR}/data_preparation.py ${EXP_DIR}/protocol.json -n 1

    # Train the DNN
    python ${CODE_DIR}/ml_localization/train.py ${CODE_DIR}/dnn/config/resnet_dropout.json --gpu none

    # Evaluate on the test set
    python ${CODE_DIR}/test.py ${EXP_DIR}/protocol.json ${CODE_DIR}/dnn/config/resnet_dropout.json
