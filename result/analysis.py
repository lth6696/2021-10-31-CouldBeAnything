import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


def style(width, height, fontsize=8):
    plt.rcParams['figure.figsize'] = (width * 0.39370, height * 0.39370)  # figure size in inches
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.sans-serif'] = 'cambria'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'


class ResultFigure:
    """
    该类用于绘制各种结果图
    """
    def __init__(self):
        pass

    # def


if __name__ == '__main__':
    results = np.load('result.npy', allow_pickle=True)
    titles = results[0]
    data = np.array([rec[1] for rec in results[1:]])
    print(pd.DataFrame(data, columns=titles))
    rf = ResultFigure()
