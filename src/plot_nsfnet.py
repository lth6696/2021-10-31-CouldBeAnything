import networkx as nx
import matplotlib.pyplot as plt

path = '../datasets/nsfnet/nsfnet.graphml'

G = nx.Graph(nx.read_graphml(path))
pos = {}
label = {}
for node in G.nodes(data=True):
    x = node[1]['positionX']
    y = node[1]['positionY']
    pos[node[0]] = (x, y)
    label[node[0]] = node[0]

nx.draw(G, pos)
nx.draw_networkx_labels(G, pos, label)
plt.show()