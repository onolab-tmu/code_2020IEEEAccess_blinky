Experiment 1: Calibration
=========================

In this experiment, we validate the calibration proposed in the paper.
We demonstrate that it is possible to acquire correct relative audio levels form the Blinkies.

Run the experiment
------------------

First, install and activate the environment using [Anaconda](https://www.anaconda.com/distribution/)

    conda env create -f environment.yml
    conda activate 2020_access_blinky_exp1
    
Thereate -n environment.yml the figures can produced by the following command

    python ./process_experiment.py 20191226_experiment_data/take_3x3

The figures are created and placed in the `figures` folder.
