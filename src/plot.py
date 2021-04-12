import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def plot_result(history, dirname='results', prefix=''):
    os.makedirs(dirname, exist_ok=True)

    for key, value in history.items():
        fig = plt.figure(figsize=(10, 5))
        plt.plot(value)
        plt.grid()
        plt.xlim(0, len(value))
        plt.title(key)
        fig.savefig(os.path.join(dirname, prefix + '_' + key + '.png'))

def plot_from_csv(filename, dirname='results'):
    os.makedirs(dirname, exist_ok=True)
    df = pd.read_csv(os.path.join(dirname, filename))
    prefix = os.path.splitext(os.path.basename(filename))[0]

    for col in df.columns:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(df[col])
        plt.grid()
        plt.xlim(0, len(df[col]))
        plt.title(col)
        fig.savefig(os.path.join(dirname, prefix + '_' + col + '.png'))

if __name__ == '__main__':
    plot_from_csv(sys.argv[1] + '.csv')
