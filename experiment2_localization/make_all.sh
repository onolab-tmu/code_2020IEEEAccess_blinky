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
python ${CODE_DIR}/ml_localization/train.py ${CODE_DIR}/dnn/config/resnet_dropout.json

# Evaluate on the test set
python ${CODE_DIR}/test.py ${EXP_DIR}/protocol.json ${CODE_DIR}/dnn/config/resnet_dropout.json
