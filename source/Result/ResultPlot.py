import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
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
    def __init__(self):
        pass

    def plot_line_graph_with_axin(self, X, y, xlabel):
        style(8.6, 6.2)
        marker = ['s', 'D']
        line_style = ['--', '-.', ':']
        y_re = [[row[i] for i in range(0, len(row), 2)] for row in y]
        fig, ax = plt.subplots(1, 1)
        for i, row in enumerate(y_re):
            plt.plot(X, row, ls=line_style[i], lw=0.5, marker=marker[i], ms=2)
        plt.xlabel(xlabel)
        plt.ylabel('Success Mapping Rate (%)')
        plt.xticks([X[i] for i in range(0, len(X), 4)])
        plt.yticks(rotation='vertical')
        plt.tight_layout()
        plt.legend(['FF-ML', 'SLF-ML'])
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        axins = inset_axes(ax, width="50%", height="40%", loc='center',
                           bbox_to_anchor=(0.2, 0, 1, 1),
                           bbox_transform=ax.transAxes,
                           axes_kwargs={'xticks': [17, 19, 21, 23, 25]})
        axins.plot([i for i in range(17, 26, 2)], [y[0][i] for i in range(8, 13, 1)], ls=line_style[0], lw=0.5, marker=marker[0], ms=2)
        axins.plot([i for i in range(17, 26, 2)], [y[1][i] for i in range(8, 13, 1)], ls=line_style[1], lw=0.5, marker=marker[1], ms=2)
        axins.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=0.5, ls=':')
        plt.show()

    def plot_line_graph(self,
                        data,
                        x: list = None,
                        xlabel: str = '',
                        ylabel: str = '',
                        fsize: tuple = (8.6, 6.2)):
        marker = ['o', 's', 'X', '*']
        line = ['-', '--', ':']
        map_vir = cm.get_cmap(name='Blues')
        solver = ['ILP-SL', 'ILP-ML', 'FF-SL', 'FF-ML', 'SLF-ML']

        if fsize is not None:
            style(*fsize)

        # 取出不同方案数据
        for i, y in enumerate(data):
            # 取出各个等级数据
            for j in range(len(y[0])):
                plt.plot([e[j] for e in y],             # set data
                         marker=marker[0], ms=2,        # set marker
                         ls=line[1], lw=0.5,            # set line
                         color=None                     # others
                         )
        plt.yticks(rotation='vertical')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    data = [[[0.3268, 0.3273, 0.3707], [0.6645, 0.6299, 0.6036], [0.9610, 0.8244, 0.8839]],
            [[1.1083, 1.0201, 1.1293], [1.1780, 1.2141, 1.1763], [1.2890, 1.2475, 1.3630]],
            [[1.4178, 1.3981, 1.2178], [1.4073, 1.5098, 1.4979], [1.4184, 1.4572, 1.5368]]]
    ResultPresentation().plot_line_graph(data)
