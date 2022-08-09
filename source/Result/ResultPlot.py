import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import json


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
        line = ['-', '--', ':', '-.']
        map_vir = cm.get_cmap(name='gist_rainbow')
        solver = ['FF-SL', 'FF-ML', 'SLF-ML']

        if fsize is not None:
            style(*fsize)

        # 取出不同方案数据
        for i, y in enumerate(data):
            # 取出各个等级数据
            for j in range(len(y[0])):
                plt.plot([e[j] for e in y],             # set data
                         marker=marker[j], ms=2,        # set marker
                         ls=line[j], lw=0.5,            # set line
                         color=map_vir(1/(len(data)*len(y[0]))*(len(y[0])*i+j))                     # others
                         )
        plt.yticks(rotation='vertical')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        plt.tight_layout()
        plt.show()

    def plot_line_graph_for_icocn(self,
                        data,
                        x: list = None,
                        xlabel: str = '',
                        ylabel: str = '',
                        fsize: tuple = (5.8, 4.4)):
        marker = ['o', 's', 'X', '^', 'p']
        line = ['-', '--', ':', '-.']
        map_vir = cm.get_cmap(name='gist_rainbow')
        solver = ['FF-SL', 'FF-ML', 'SLF-ML']

        if fsize is not None:
            style(*fsize)

        # 取出不同方案数据
        for i, y in enumerate(data):
            plt.plot(x,
                     y,                             # set data
                     marker=marker[i], ms=2,        # set marker
                     ls=line[int(i%len(line))], lw=0.5,            # set line
                     color=map_vir((i+1)/len(data)), # others
                     label=solver[i]
                     )
        plt.yticks(rotation='vertical')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.xticks([i*5 for i in range(3)])
        plt.yticks([i*4 for i in range(0, 5)])
        plt.axhline(y=16, xmin=0.045, xmax=0.955, ls='-.', lw=1, color='#8B0000', label='Network Capacity')
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    result_json = json.load(open('../results_for_sci.json'))
    var = 'network_capacity'
    # ResultPresentation().plot_line_graph_for_icocn([[np.sum(row) for row in result_json[solver][var]] for solver in ['FF-SL', 'FF-ML', 'SLF-ML']],
    # ResultPresentation().plot_line_graph_for_icocn([[row[2] for row in result_json[solver][var]] for solver in ['FF-SL', 'FF-ML', 'SLF-ML']],
    ResultPresentation().plot_line_graph_for_icocn([result_json[solver][var] for solver in ['FF-SL', 'FF-ML', 'SLF-ML']],
                                                   x=[i for i in range(1, 41)],
                                                   xlabel='Number of Traffic Matrix',
                                                   ylabel='Throughput (Tb/s)',
                                                   fsize=(8.6, 6.2)
                                                   # fsize=(6, 4.5)
                                                   )
