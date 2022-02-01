import networkx as nx
import matplotlib.pyplot as plt
from plot_style import Style


def hexnet():
    path = '../datasets/hexnet/hexnet.graphml'

    G = nx.Graph(nx.read_graphml(path))
    pos = {}
    label = {}
    for node in G.nodes(data=True):
        x = node[1]['positionX']
        y = node[1]['positionY']
        pos[node[0]] = (x, y)
        label[node[0]] = node[0]

    Style.style(width=8.6, height=4)
    nx.draw(G, pos, node_size=100, node_color='#FADEC2',
            linewidths=0.5, edgecolors='#42454A',
            width=1, edge_color='#AEB9CA')
    nx.draw_networkx_labels(G, pos, label, font_family='serif', font_size=8)
    plt.show()


def nsfnet():
    path = '../datasets/nsfnet/nsfnet.graphml'

    G = nx.Graph(nx.read_graphml(path))
    pos = {}
    label = {}
    for node in G.nodes(data=True):
        x = node[1]['positionX']
        y = node[1]['positionY']
        pos[node[0]] = (x, -y)
        label[node[0]] = node[0]

    Style.style(width=8.6, height=4)
    nx.draw(G, pos, node_size=100, node_color='#FADEC2',
            linewidths=0.5, edgecolors='#42454A',
            width=1, edge_color='#AEB9CA')
    nx.draw_networkx_labels(G, pos, label, font_family='serif', font_size=8)
    plt.show()


def geant2():
    path = '../datasets/geant2/geant2.graphml'

    G = nx.Graph(nx.read_graphml(path))
    pos = {}
    label = {}
    for node in G.nodes(data=True):
        x = node[1]['positionX']
        y = node[1]['positionY']
        pos[node[0]] = (x, -y)
        label[node[0]] = node[0]

    Style.style(width=8.6, height=4)
    nx.draw(G, pos, node_size=100, node_color='#FADEC2',
            linewidths=0.5, edgecolors='#42454A',
            width=1, edge_color='#AEB9CA')
    nx.draw_networkx_labels(G, pos, label, font_family='serif', font_size=8)
    plt.show()


if __name__ == '__main__':
    geant2()