import matplotlib.pyplot as plt
from matplotlib import cm
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

    def plot_line(self, y):
        style(13, 7)
        plt.plot(y)
        plt.show()

    def plot_multi_lines(self, y):
        for row in y:
            plt.plot(row)
        plt.show()

    def plot_block_distribution_under_different_situation(self, y, width=0.8, situations=None):
        y = [col[1:] for col in y]
        # X = [0, (len(y) + 5) * width]
        X = [(len(y) + 5) * width * i for i in range(len(y[0]))]
        map_vir = cm.get_cmap(name='Blues')
        for i, record in enumerate(y):
            plt.bar([j+width*i for j in X], record, width=width, color=map_vir((len(y)-i)/len(y)))
        if situations is None:
            situations = ['0x{}'.format(str(i+1).zfill(2)) for i in range(len(X))]
        plt.xticks([x+(len(y)*width)/2 for x in X], situations)
        plt.show()

    def plot_hist(self, y):
        plt.hist(y)
        plt.show()


if __name__ == '__main__':
    t = ResultPresentation()
    x = [[1.0, 0.9615384615384616, 7.142857142857142], [1.5, 3.7774725274725274, 3.5714285714285716],
         [2.0, 3.296703296703297, 1.4652014652014653], [2.5, 2.8159340659340657, 1.9230769230769231],
         [3.0, 3.021978021978022, 0.4945054945054945], [3.8333333333333335, 1.6025641025641024, 0.8241758241758244],
         [3.5, 2.0375457875457874, 0.38919413919413914], [4.75, 1.510989010989011, 0.4635989010989011],
         [6.666666666666667, 1.0531135531135531, 0.5837912087912088], [6.0, 1.3236763236763238, 0.1873126873126873],
         [6.583333333333333, 1.0645604395604396, 0.3663003663003663], [7.0, 1.3947590870667794, 0.23245984784446322],
         [8.0, 0.9157509157509158, 0.4853479853479854], [6.0, 1.2737262737262738, 0.2997002997002997],
         [5.5, 1.1813186813186813, 0.23351648351648352], [5.5, 1.043956043956044, 0.28846153846153844],
         [8.133333333333333, 1.1080586080586086, 0.12820512820512822],
         [10.222222222222221, 0.6791819291819292, 0.48840048840048844],
         [7.642857142857143, 1.1773940345368916, 0.2845368916797488]]
    t.plot_block_distribution_under_different_situation(x)
