import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import networkx as nx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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

    def plot_topology(self, name: str, width: float = 8.6, height: float = 5):
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


if __name__ == '__main__':
    # ILP vs Heuristic
    # y = [[98.0, 96.2, 93.1, 92.0],
    #      [92.3, 90.0, 86.6, 80.8],
    #      [96.98, 95.18, 92.01, 86.18],
    #      [97.08, 95.42, 92.82, 85.33],
    #      [91.82, 88.07, 81.51, 76.19]]

    solutions = (
        'SASMA-LSMS', 'SASMA-LFEL', 'SASMA-EO'
        # 'EO-(0.3,0.3,0.3)', 'EO-(0.5,0.5,0)', 'EO-(0.5,0,0.5)', 'EO-(0,0.5,0.5)', 'EO-(1,0,0)', 'EO-(0,0,1)'
    )
    performance_metrics = ('mapping_rate', 'service_throughput',
                           'network_throughput', 'req_bandwidth',
                           'ave_hops', 'ave_link_utilization',
                           'ave_level_deviation')
    K = 25
    metrics = 'ave_level_deviation'
    all_data = np.zeros(shape=(len(solutions), K))
    for i, name in enumerate(solutions):
        if not os.path.exists('../{}.npy'.format(name)):
            continue
        solution_data = np.load('../{}.npy'.format(name))
        all_data[i] = [np.mean(k, axis=0)[performance_metrics.index(metrics)] for k in solution_data]
    plt = ResultPresentation(solutions).plot_line_figure(
        np.array([i + 1 for i in range(K)]),
        all_data,
        'Number of traffic matrices',
        'Level Deviation',
        show=True
    )
    # ResultPresentation(solutions).set_baseline(plt, [i+1 for i in range(K)], [16 for _ in range(K)], name='Max Throughput')
