import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


class ResultFigure:
    """
    该类用于绘制各种结果图
    """

    def __init__(self):
        # 画图属性
        self.markers = ['o', 's', 'X', '*']
        self.line_styles = ['-', '--', ':', '-.']
        self.map_vir = cm.get_cmap(name='gist_rainbow')

    @staticmethod
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

    def plot_line(
            self,
            data: np.array,
            X: np.array,
            xlabel: str = None,
            ylabel: str = None,
            fsize: tuple = (8.6, 6.2),
            ms: float = 2.0,
            lw: float = 0.5,
            show: bool = True,
            label: list = None
    ):
        # 设置图片大小
        self.style(*fsize)
        # 获取数据大小
        shape = data.shape
        if len(shape) > 2:
            raise ValueError('The size of data {} is out of 2D.'.format(shape))
        elif len(shape) == 2:
            # 取出不同方案数据
            for i, record in enumerate(data):
                plt.plot(
                    X, record,  # set data
                    marker=self.markers[i % len(self.markers)], ms=ms,  # set marker
                    ls=self.line_styles[i % len(self.line_styles)], lw=lw,  # set line
                    color=self.map_vir(i / shape[0])  # set color
                )
        else:
            plt.plot(
                X, data,
                marker=self.markers[0], ms=ms,  # set marker
                ls=self.line_styles[0], lw=lw,  # set line
            )

        plt.yticks(rotation='vertical')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, ls=':', lw=lw, c='#d5d6d8')
        plt.tight_layout()
        plt.legend(label)

        if show:
            plt.show()
        return plt


if __name__ == '__main__':
    results_nsga = np.load('moea_psy_NSGA2_.npy', allow_pickle=True)
    results_cara = np.load('cara.npy', allow_pickle=True)
    titles = results_nsga[0]
    data_nsga = np.array([rec[1] for rec in results_nsga[1:]])
    data_cara = np.array([rec[1] for rec in results_cara[1:]])
    rf = ResultFigure()
    """
    'latency(s)',   'hop',              'routed services', 
    'success rate', 'throughput',       'com_utl', 
    'sto_utl',      'bandwidth_utl',    'cost'
    """
    cost_compare = [
        [267.31084623, 209.57156449, 169.7536106, 155.47256394, 136.4919122],  # nsga
        [328.40368889, 221.09437341, 173.04075263, 166.36455624, 165.65425288]  # cara
    ]
    index = 6
    rf.plot_line(
        np.array([data_nsga[:, index], data_cara[:, index]]),
        # np.array(cost_compare),
        [i + 1 for i in range(data_nsga.shape[0])],
        xlabel='Number of traffic matrices',
        ylabel='Link Utilization (%)',
        show=False,
        label=['NSGA-II', 'Dijkstra']
    )
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    all_data = np.array([data_nsga[:, index], data_cara[:, index]])
    # all_data = np.array(cost_compare)
    a = np.max(all_data, axis=1)
    b = np.min(all_data, axis=1)
    c = np.mean(all_data, axis=1)
    d = np.mean(all_data[0, :] - all_data[1, :])
    print(pd.DataFrame([a, b, c],
                       columns=['NSGA', 'Dijkstra'],
                       index=['max', 'min', 'average']))
    print(all_data)
