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

    def plot_heuristic_and_ilp_for_success_rate(self, X, y, xlabel):
        style(8.6, 6.2)
        marker = ['o', 's', 'D']
        line_style = ['-', '--', '-.', ':']
        for i, row in enumerate(y):
            plt.plot(X, row, ls=line_style[i], lw=0.5, marker=marker[i], ms=2)
        plt.xlabel(xlabel)
        plt.ylabel('Success Mapping Rate (%)')
        plt.xticks([i for i in X])
        plt.yticks(rotation='vertical')
        plt.tight_layout()
        plt.legend(['ILP-ML', 'FF-ML', 'SLF-ML'])
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        plt.show()

    def plot_heuristic_for_success_rate(self, X, y, xlabel):
        style(8.6, 6.2)
        marker = ['s', 'D']
        line_style = ['--', '-.', ':']
        y = [[row[i] for i in range(0, len(row), 2)] for row in y]
        for i, row in enumerate(y):
            plt.plot(X, row, ls=line_style[i], lw=0.5, marker=marker[i], ms=2)
        plt.xlabel(xlabel)
        plt.ylabel('Success Mapping Rate (%)')
        plt.xticks([X[i] for i in range(0, len(X), 2)])
        plt.yticks(rotation='vertical')
        plt.tight_layout()
        plt.legend(['FF-ML', 'SLF-ML'])
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        plt.show()

    def plot_heuristic_for_success_rate_with_axin(self, X, y, xlabel):
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

    def plot_block_distribution_under_different_situation(self, y, width=0.8, situations=None):
        style(8.6, 6.2)
        y = [col[1:] for col in y]
        X = [(len(y) + 5) * width * i for i in range(len(y[0]))]
        map_vir = cm.get_cmap(name='spring')
        for i, record in enumerate(y):
            plt.bar([j+width*i for j in X], record, width=width, color=map_vir((len(y)-i)/len(y)))
        if situations is None:
            situations = ['0x{}'.format(str(i+1).zfill(2)) for i in range(len(X))]
        # plt.xticks([x+(len(y)*width)/2 for x in X], situations)
        plt.xticks([x + (len(y) * width) / 2 for x in X], ['Path', 'Bandwidth'])
        plt.yticks([i for i in range(0, 81, 20)], rotation='vertical')
        plt.xlabel('Different Situation')
        plt.ylabel('Sum Blocking Probability (%)')
        plt.tight_layout()
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        plt.show()

    def plot_link_utilization_per_level(self, y, width=0.8):
        style(8.6, 6.2)
        map_vir = cm.get_cmap(name='spring')
        X = [(len(y) + 5) * width * i for i in range(len(y[0]))]
        for i, record in enumerate(y):
            plt.bar([j+width*i for j in X], record, width=width, color=map_vir((len(y)-i)/len(y)))
        plt.yticks([i/10 for i in range(0, 11, 2)])
        plt.show()

    def plot_link_utilization(self, y):
        style(8.6, 6.2)
        marker = ['o', 's', 'D']
        line_style = ['--', ':', '-']
        map_vir = cm.get_cmap(name='Blues')
        color = ['#ff7f0e', '#2ca02c']
        X = [i for i in range(1, len(y[0])*2+1, 4)]
        y_re = [[row[i] for i in range(0, len(row), 2)] for row in y]
        for i, row in enumerate(y_re):
            plt.plot(X, row, ls=line_style[i%3], lw=0.5, marker=marker[i%3], ms=2, color=color[int(i/3)])
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Number of Traffic Matrix')
        plt.ylabel('Level Deviation')
        # plt.xticks([X[i] for i in range(0, len(X), 4)])
        plt.yticks([i/10 for i in range(0, 11, 2)], rotation='vertical')
        plt.tight_layout()
        plt.legend(['FF-ML-L1', 'FF-ML-L2', 'FF-ML-L3', 'SLF-ML-L1', 'SLF-ML-L2', 'SLF-ML-L3'], ncol=2)
        plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
        plt.show()

    def plot_hist(self, y):
        plt.hist(y)
        plt.show()
