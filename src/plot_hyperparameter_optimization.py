# import seaborn as sns
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# plt.style.use('dark_background')

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from pprint import pprint
from itertools import combinations
import math


def load_trials(model_filename):
    model_path = Path(model_filename)

    if model_path.is_file():
        print('Found saved Trials! Loading...')
        # Load an already saved trials object, and increase the max
        with open(model_filename, 'rb') as f:
            trials = pickle.load(f)

        return trials

    else:
        print('Error: file {} not found'.format(model_filename))
        exit()


def trials_to_dataframe(trials):
    data = []
    for trial in trials.trials:
        d = {}
        d['loss'] = trial['result']['loss']
        item = trial['misc']['vals']
        for k, v in item.items():
            d[k] = v[0]
        data.append(d)
    
    trials_df = pd.DataFrame(data)

    # print(trials_df.info())

    return trials_df


def print_best(trials_df):
    print('Number of trials: {}'.format(trials_df.shape[0]))

    print('Top 10 scores:')
    print('=============')
    print(trials_df.sort_values(by='loss').head(10))


def plot_hyperparameter_optimization(trials_df):
    parameters = [col for col in trials_df.columns if col != 'loss']
    cols = len(parameters)

    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(15,9))
    axes = axes.flatten()
    cmap = plt.cm.hsv
    for i, val in enumerate(parameters):
        x = trials_df[val]
        y = trials_df['loss']
        axes[i].scatter(x, y, s=20, linewidth=0.01, alpha=0.9, c=cmap(float(i)/cols))
        axes[i].set_title('loss vs. {}'.format(val))
        axes[i].set_xlabel(val)
        axes[i].set_ylabel('loss')
        axes[i].set_facecolor("black")

    plt.tight_layout()
    plt.show()


def plot_parameter_dependencies(trials_df):
    parameters = [col for col in trials_df.columns if col != 'loss']

    param_combos = [item[::-1] for item in combinations(parameters, 2)]
    num_combos = len(param_combos)

    rows = int(math.sqrt(num_combos))
    if num_combos % rows == 0:
        cols = num_combos // rows
    else:
        cols = num_combos // rows + 1

    # print(param_combos)

    f, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,9))
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, param_combo in enumerate(param_combos):
        x = trials_df[param_combo[0]]
        y = trials_df[param_combo[1]]
        c = trials_df['loss']
        axes[i].scatter(x, y, s=20, linewidth=0.01, alpha=0.9, c=c, cmap='gnuplot2_r')
        axes[i].set_title('{} vs. {}'.format(param_combo[1], param_combo[0]))
        axes[i].set_xlabel(param_combo[0])
        axes[i].set_ylabel(param_combo[1])
        axes[i].set_facecolor("black")

    plt.tight_layout()
    plt.show()


def main():

    ratings_filename = '../data/ratings_ugt9_igt9'
    suffix = '_unadj_nobias'
    model_filename = '{}{}.hyperopt'.format(ratings_filename, suffix)

    trials = load_trials(model_filename)
    trials_df = trials_to_dataframe(trials)
    print_best(trials_df)
    plot_hyperparameter_optimization(trials_df)
    plot_parameter_dependencies(trials_df)


if __name__ == '__main__':
    main()