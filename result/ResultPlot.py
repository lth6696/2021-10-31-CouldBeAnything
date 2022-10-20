import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib import cm
import networkx as nx


def style(width, height, fontsize=8):
    plt.rcParams['figure.figsize'] = (width * 0.39370, height * 0.39370)  # figure size in inches
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.sans-serif'] = 'cambria'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'


class TopologyPresentation(object):
    def __init__(self):
        self.path = {'nsfnet': '../graphml/nsfnet/nsfnet.graphml',
                     'hexnet': '../graphml/hexnet/hexnet.graphml',
                     'geant2': '../graphml/geant2/geant2.graphml'}

    def plot_topology(self, name: str, width: float = 8.6, height: float = 6):
        if name not in list(self.path.keys()):
            raise Exception('Wrong name \'{}\''.format(name))
        G = nx.Graph(nx.read_graphml(self.path[name]))
        pos = {}
        label = {}
        for node in G.nodes(data=True):
            x = node[1]['positionX']
            y = node[1]['positionY']
            pos[node[0]] = (x, y)
            label[node[0]] = node[0]

        style(width=width, height=height)
        nx.draw(G, pos, node_size=100, node_color='#FADEC2',
                linewidths=0.5, edgecolors='#42454A',
                width=1, edge_color='#AEB9CA')
        nx.draw_networkx_labels(G, pos, label, font_family='serif', font_size=8)
        plt.show()


class ResultPresentation(object):
    def __init__(self, solutions):
        # 画图属性
        self.markers = ['o', 's', 'X', '*']
        self.line_styles = ['-', '--', ':', '-.']
        self.map_vir = cm.get_cmap(name='gist_rainbow')

        # 仿真属性
        self.solutions = solutions

    def plot_line_figure(
            self,
            X: np.ndarray,
            data: np.ndarray,
            xlabel: str = None,
            ylabel: str = None,
            fsize: tuple = (8.6, 6.2),
            ms: float = 2.0,
            lw: float = 0.5,
            show: bool = True
    ):
        # 设置图片大小
        style(*fsize)
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
        plt.legend(self.solutions)

        if show:
            plt.show()
        return plt

    def set_baseline(
            self,
            plt,
            x,
            y,
            name: str = None,
            m: str = 'o',       # marker
            ms: float = 2,      # markersize
            mew: float = 0.5,   # markeredgewidth
            mfc: str = 'white', # markerfacecolor
            ls: str = '-',      # linestyle
            lw: float = 0.5,    # linewidth
            c: str = 'black'    # color
    ):
        plt.plot(
            x, y,
            marker=m,
            markersize=ms,
            markeredgewidth=mew,
            markerfacecolor=mfc,
            linestyle=ls,
            linewidth=lw,
            color=c
        )
        plt.legend([*self.solutions, name] if name else self.solutions)
        plt.show()

    def get_derivation(
            self,
            X: np.ndarray,
            data: np.ndarray,
            fsize: tuple = (8.6, 6.2),
            xlabel: str = None,
            ylabel: list = None,
            ms: float = 2.0,
            lw: float = 0.5,
            show: bool = True
    ):
        shape = data.shape
        if len(shape) != 2:
            raise ValueError
        # 初始化子图
        style(*fsize)
        grid = plt.GridSpec(3, 1, wspace=0.5, hspace=0.5)
        ax_solutions = plt.subplot(grid[0:2, 0], xticklabels=[])
        for i, record in enumerate(data):
            plt.plot(
                X, record,  # set data
                marker=self.markers[i % len(self.markers)], ms=ms,  # set marker
                ls=self.line_styles[i % len(self.line_styles)], lw=lw,  # set line
                color=self.map_vir(i / shape[0])  # set color
            )
        plt.plot(
            X, [16 for _ in range(shape[1])],
            marker='o',
            markersize=2,
            markeredgewidth=0.5,
            markerfacecolor='white',
            linestyle='-',
            linewidth=0.5,
            color='black'
        )
        # plt.axvline(x=6, ymax=0.69, ls='--', lw=0.5, color='black')
        plt.yticks(rotation='vertical')
        plt.ylabel(ylabel[0])
        plt.xticks([i * 5 for i in range(6)])
        plt.yticks([i * 4 for i in range(5)])
        plt.grid(True, ls=':', lw=lw, c='#d5d6d8')
        plt.tight_layout()
        plt.legend([*self.solutions, 'Max Throughput'])

        ax_derivation = plt.subplot(grid[2, 0])
        derivation = np.zeros(shape=shape)
        for i, record in enumerate(data):
            dfunc = np.poly1d(np.polyfit(X, record, deg=5)).deriv()
            derivation[i] = dfunc(X)
            plt.plot(
                X, derivation[i],
                marker=self.markers[i % len(self.markers)], ms=ms,  # set marker
                ls=self.line_styles[i % len(self.line_styles)], lw=lw,  # set line
                color=self.map_vir(i / shape[0])  # set color
            )
        plt.axvline(x=6, ymax=0.34, ls='--', lw=0.5, color='black')
        plt.axhline(y=1, xmax=0.235, ls='--', lw=0.5, color='black')
        plt.xticks([i * 5 for i in range(6)])
        plt.yticks([i for i in range(4)])
        plt.yticks(rotation='vertical')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel[1])
        plt.grid(True, ls=':', lw=lw, c='#d5d6d8')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # TopologyPresentation().plot_topology('hexnet')
    solutions = (
        'ILP-LBMS', 'ILP-LSMS', 'LFEL-LBMS', 'LFEL-LSMS', 'EO-LBMS', 'EO-LSMS'
        # 'EO-LBMS-(0.3,0.3,0.3)', 'EO-LBMS-(0.5,0.5,0)', 'EO-LBMS-(0.5,0,0.5)', 'EO-LBMS-(0,0.5,0.5)', 'EO-LBMS-(1,0,0)', 'EO-LBMS-(0,0,1)',
        # 'EO-LSMS-(0,0.5,0.5)', 'EO-LSMS-(0,0,1)'
        # 'LFEL-LBMS', 'LFEL-LSMS', 'EO-LBMS', 'EO-LSMS'
    )
    performance_metrics = ('mapping_rate', 'service_throughput',
                           'network_throughput', 'req_bandwidth',
                           'ave_hops', 'ave_link_utilization',
                           'ave_level_deviation')
    K = 25
    metrics = 'mapping_rate'
    ILP_vs_SASMA_data = np.array([
        [98.8, 96.2, 93.1, 91.4],   # ILP-LBMS
        [94.3, 90.2, 85.8, 80.4],   # ILP-LSMS
        [98.5, 94.8, 90.1, 86.0],   # LFEL-LBMS
        [94.0, 87.0, 81.0, 75.9],   # LFEL-LSMS
        [98.3, 95.1, 91.5, 87.7],   # EO-LBMS
        [94.2, 87.4, 82.1, 76.0]    # EO-LSMS
    ])
    X = np.array([i + 1 for i in range(K)])
    all_data = np.zeros(shape=(len(solutions), K))
    for i, name in enumerate(solutions):
        if not os.path.exists('../save_files/HEXNET-SASMA/{}.npy'.format(name)):
            continue
        solution_data = np.load('../save_files/HEXNET-SASMA/{}.npy'.format(name))
        all_data[i] = [np.mean(k, axis=0)[performance_metrics.index(metrics)] * 100 for k in solution_data]
    plt = ResultPresentation(solutions).plot_line_figure(
        # X,
        np.array([4, 8, 12, 16]),
        # all_data[:, np.arange(3, 16, 4)],
        ILP_vs_SASMA_data,
        'Number of traffic matrices',
        'Mapping Rate (%)',
        show=True
    )
    # plt.axvline(x=18, ymax=0.69, ls='--', lw=0.5, color='black')
    # plt.axhline(y=100, xmax=0.69, ls='--', lw=0.5, color='black')
    # plt.show()
    # ResultPresentation(solutions).get_derivation(
    #     X, all_data,
    #     xlabel='Number of traffic matrices',
    #     ylabel=['Throughput (Tb/s)', 'Growth Rate']
    # )
    pandas.set_option('display.max_columns', 100)
    pandas.set_option('display.width', 1000)
    a = np.max(all_data, axis=1)
    b = np.min(all_data, axis=1)
    # c = np.mean(all_data[:, np.arange(0, 5)], axis=1)
    d = np.mean(all_data[:, np.arange(0, 17)], axis=1)
    e = np.mean(all_data[:, np.arange(17, 24)], axis=1)
    print(pandas.DataFrame([solutions, a, b, d, e]))
