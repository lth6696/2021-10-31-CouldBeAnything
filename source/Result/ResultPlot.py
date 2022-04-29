import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def style(width, height, fontsize=8):
    plt.rcParams['figure.figsize'] = (width * 0.39370, height * 0.39370)  # figure size in inches
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
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

    def plot_line(self, x, y=0):
        style(13, 7)
        plt.plot(x)
        plt.show()


if __name__ == '__main__':
    t = TopologyPresentation()
    t.plot_topology('nsfnet')