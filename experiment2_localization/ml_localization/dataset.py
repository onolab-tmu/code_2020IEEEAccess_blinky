import os, json
import jsongzip
import numpy as np
from scipy.io import wavfile

def get_formatters(method='reshape', frames=(1,16), outputs=(0,2)):
    '''
    Return the data formating function

    Parameters
    ----------
    method: str
        Either 'reshape' or 'avg'
    frames: int or slice
        The frames to use
    outputs: int or slice
        The outputs to use
    '''

    if method == 'reshape':

        def data_formatter(e):
            return np.array(e, dtype=np.float32)[slice(*frames),:].reshape((1,-1))

    elif method == 'avg':

        def data_formatter(e):
            return np.array(e, dtype=np.float32)[slice(*frames),:].mean(axis=0, keepdims=True)

    elif method == 'none':

        def data_formatter(e):
            return np.array(e, dtype=np.float32)

    def label_formatter(l):
        return np.array(l[slice(*outputs)], dtype=np.float32)

    def skip(e):
        all_finite = np.all(np.isfinite(e[0])) and np.all(np.isfinite(e[1]))
        return not (all_finite and np.all(e[0] < 1000.))

    return data_formatter, label_formatter, skip


def get_data_raw(metadata_file, data_formatter=None, label_formatter=None, skip=None):

    data_list = jsongzip.load(metadata_file)

    def format_examples(examples):
        formated_examples = []
        for example in examples:

            if data_formatter is not None:
                example[0] = data_formatter(example[0])

            if label_formatter is not None:
                example[1] = label_formatter(example[1])

            if skip(example):
                print('skip nan')
                continue

            formated_examples.append(tuple(example))

        return formated_examples

    processed_datasets = dict()

    for name, data in data_list.items():
        processed_datasets[name] = format_examples(data)

    return processed_datasets


def get_data(metadata_file, data_formatter=None, label_formatter=None, skip=None):

    proc_data = get_data_raw(
            metadata_file,
            data_formatter=data_formatter,
            label_formatter=label_formatter,
            skip=skip,
            )

    return proc_data['train'], proc_data['validation'], proc_data['test']

