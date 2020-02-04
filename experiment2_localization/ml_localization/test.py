import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import pandas as pd
import chainer
import seaborn as sns
import matplotlib.pyplot as plt

from ml_localization import get_data, models, get_formatters

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run a model on a test set')
    parser.add_argument('config', type=str, help='The JSON file containing the configuration.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # import model and use MSE
    nn = models[config['model']['name']](*config['model']['args'], **config['model']['kwargs'])
    chainer.serializers.load_npz(config['model']['file'], nn)

    # Helper to load the dataset
    config['data']['format_kwargs'].pop('outputs')
    data_formatter, label_formatter, skip = get_formatters(outputs=(0,4), **config['data']['format_kwargs'])

    # Load the dataset
    train, validate, test = get_data(config['data']['file'],
            data_formatter=data_formatter, 
            label_formatter=label_formatter, skip=skip)

    table = []

    for (example, label) in train:
        # get all samples with the correct noise variance
        table.append([
                    np.log10(np.sqrt(np.linalg.norm(label[:2] - np.squeeze(nn(example).data))**2)), 
                    label[2],
                    'train',  # noise variance,
                    ])

    for (example, label) in validate:
        # get all samples with the correct noise variance
        table.append([
                    np.log10(np.sqrt(np.linalg.norm(label[:2] - np.squeeze(nn(example).data))**2)), 
                    label[2],
                    '$-\infty$',  # noise variance,
                    ])

    for (example, label) in test:
        # get all samples with the correct noise variance
        table.append([
                    np.log10(np.sqrt(np.linalg.norm(label[:2] - np.squeeze(nn(example).data))**2)), 
                    label[2],
                    np.log10(label[3]),  # noise variance,
                    ])

    df = pd.DataFrame(data=table, columns=['RMSE (log10)', 'Gain [dB]', 'Noise variance (log10)'])

    sns.violinplot(data=df, x='Noise variance (log10)', y='RMSE (log10)')
    #plt.ylim([-2.5, 1])
    plt.savefig('mse.pdf')
