import matplotlib
matplotlib.use('Agg')

import argparse
import json, os
import numpy as np
import pandas as pd
import chainer

import seaborn as sns
import matplotlib.pyplot as plt

from ml_localization import get_data_raw, get_data, models, get_formatters

def baseline(blinky_signals, blinky_locations, k_max=1):

    I = np.argsort(blinky_signals, axis=1)[:,-k_max:]

    locs = []
    weights = []
    for i in range(k_max):
        ind = (np.arange(blinky_signals.shape[0]), I[:,i])
        weights.append(blinky_signals[ind][:,None])
        locs.append(blinky_locations[I[:,i],None,:])

    weights = np.concatenate(weights, axis=1)
    locs = np.concatenate(locs, axis=1)

    # create weights
    weights /= np.sum(weights, axis=1, keepdims=True)

    # weighted combinations of blinky locations
    est = np.sum(weights[:,:,None] * locs, axis=1)

    return est


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a model on a test set')
    parser.add_argument('protocol', type=str,
            help='The JSON file containing the experimental details.')
    parser.add_argument('configs', metavar='CONFIG', nargs='+', type=str,
            help='JSON files containing network configurations')
    args = parser.parse_args()


    with open(args.protocol, 'r') as f:
        protocol = json.load(f)

    blinky_locations = np.array(protocol['blinky_locations'])

    dataset_file = None
    algorithms = dict(
            baseline1=dict(f=baseline, args=[blinky_locations], kwargs=dict(k_max=1)),
            baseline2=dict(f=baseline, args=[blinky_locations], kwargs=dict(k_max=2)),
            baseline3=dict(f=baseline, args=[blinky_locations], kwargs=dict(k_max=3)),
            )
    
    for config_fn in args.configs:

        with open(config_fn, 'r') as f:
            config = json.load(f)

        # import model and use MSE
        nn = models[config['model']['name']](*config['model']['args'], **config['model']['kwargs'])
        chainer.serializers.load_npz(config['model']['file'], nn)

        def nn_wrap(x):
            with chainer.using_config('train', False):
                y = nn(x).data
            return y

        algorithms[config['name']] = dict(f=nn_wrap, args=[], kwargs={})

        # Helper to load the dataset
        if 'outputs' in config['data']['format_kwargs']:
            config['data']['format_kwargs'].pop('outputs')
        data_formatter, label_formatter, skip = get_formatters(outputs=(0,2), **config['data']['format_kwargs'])

        if dataset_file is None:
            dataset_file = '/Volumes' + config['data']['file']

            # Load the dataset
            sets = get_data_raw(dataset_file,
                    data_formatter=data_formatter, 
                    label_formatter=label_formatter, skip=skip)

        elif dataset_file != config['data']['file']:

            raise ValueError('The model {} was trained with a different dataset!')

    # Run all the algorithms and store in a dataframe
    columns = ['Set', 'Algorithm', 'Error', 'x', 'y']
    df = pd.DataFrame(columns=columns)

    for set_name, data in sets.items():

        examples, labels = np.array([e[0] for e in data]), np.array([e[1] for e in data])

        for alg_name, alg in algorithms.items():

            est_loc = alg['f'](examples, *alg['args'], **alg['kwargs'])
            err_vec = est_loc - labels
            error = np.linalg.norm(err_vec, axis=1)

            df_tmp = pd.concat((
                        pd.DataFrame([set_name] * len(examples), columns=columns[:1]),
                        pd.DataFrame([alg_name] * len(examples), columns=columns[1:2]),
                        pd.DataFrame(error, columns=columns[2:3]),
                        pd.DataFrame(err_vec, columns=columns[3:]),
                        ), axis=1)
            df = df.append(df_tmp)

    # Print out some stats
    p50 = pd.pivot_table(
            df, columns=['Set', 'Algorithm'], values=['Error'],
            aggfunc=lambda x:np.percentile(x, 50))
    p90 = pd.pivot_table(
            df, columns=['Set', 'Algorithm'], values=['Error'],
            aggfunc=lambda x:np.percentile(x, 90))
    print(p50)
    print(p90)

    # Create the plots

    sns.set(style='whitegrid', context='paper', 
            #palette=sns.light_palette('navy', n_colors=2),
            #palette=sns.light_palette((210, 90, 60), input="husl", n_colors=2),
            font_scale=0.9,
            rc={
                'figure.figsize':(3.38649 * 0.8, 3.338649 / 3), 
                'lines.linewidth':1.,
                #'font.family': u'Roboto',
                #'font.sans-serif': [u'Roboto Bold'],
                'text.usetex': False,
                })

    sns.boxplot(data=df, x='Set', y='Error', hue='Algorithm')
    sns.factorplot(data=df, x='Algorithm', y='Error', col='Set', kind='box')
    plt.ylim([0, 200])
    plt.savefig('pub_2018_ASJ_fall/figures/mse.pdf')

    '''
    # 90th percentile in L_inf norm (for test set)
    df_test = df[df['Set'] == 'test']
    p90 = np.percentile(np.maximum(np.abs(df_test['x']), np.abs(df_test['y'])), 99)
    blim = [-p90, p90]

    # first we find the 90 percentile (circularly)
    for algo in ['nn', 'baseline1', 'baseline2', 'baseline3']:
        I = np.logical_and(df['Set'] == 'test', df['Algorithm'] == algo)
        sns.jointplot(x='x', y='y', data=df[I],
                kind='scatter', stat_func=None,
                xlim=blim, ylim=blim, s=2)
        plt.savefig('pub_2018_ASJ_fall/scatter_{}_{}.pdf'.format(config['name'], algo))
    '''

    plt.show()
