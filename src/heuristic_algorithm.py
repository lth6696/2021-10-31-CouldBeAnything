import networkx as nx
import matplotlib.pyplot as plt
import random


# generate traffic matrix
def gen_traffic_matrix(nodes):
    random.seed(0)


# generate lightpath topology
def gen_lightpath_topo():
    # G = nx.read_graphml('../datasets/nsfnet/nsfnet.graphml')
    G = nx.read_graphml('nsfnet.graphml')
    pos = {}
    for node in G.nodes(data=True):
        x = node[1]['positionX']
        y = node[1]['positionY']
        pos[node[0]] = (x, y)
    nx.draw(G, pos)
    plt.show()


gen_traffic_matrix(9)
gen_lightpath_topo()