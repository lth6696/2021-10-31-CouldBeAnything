import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

pd.set_option('display.max_column', 200)
pd.set_option('display.width', 200)


# generate traffic matrix
def gen_traffic_matrix(nodes, taf):
    random.seed(0)
    traffic_matrix = []
    for i in range(nodes):
        tmp_i = []
        for j in range(nodes):
            if i != j:
                tmp_i.append((random.randint(0, 10), taf[random.randint(0, len(taf)-1)]))
            else:
                tmp_i.append((0, 0))
        traffic_matrix.append(tmp_i)
    return traffic_matrix


# generate lightpath topology
def gen_lightpath_topo(path, wavelengths, draw):
    random.seed(0)
    G = nx.MultiGraph(nx.read_graphml(path))
    pos = {}
    label = {}
    for node in G.nodes(data=True):
        x = node[1]['positionX']
        y = node[1]['positionY']
        pos[node[0]] = (x, y)
        label[node[0]] = node[0]

    adj_matrix = np.array(nx.adjacency_matrix(G).todense())
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix)):
            adj_matrix[row][col] = adj_matrix[row][col] * random.randint(1, wavelengths)
            for i in range(adj_matrix[row][col] - 1):
                G.add_edge(row, col, id='{:02d}{:02d}{:02d}'.format(row, col, i+1))

    if draw:
        nx.draw(G, pos)
        nx.draw_networkx_labels(G, pos, label)
        plt.show()

    return adj_matrix, G


# Find a path to carry service
def get_property_path(G, service, tafs):
    """
    1 - 按单跳至多跳找多条路径，其中，因子要连续taf
    2 - 遍历寻到的路径，分配资源。若有冗余，则分配；反之，继续遍历
    """
    data_res = service[0]
    t = service[1]
    loss = []
    if data_res:
        tafs.sort()     # 升序
        t_pos = tafs.index(t)
        


# A heuristic algorithm
def heuristic_algorithm():
    path = '../datasets/nsfnet/nsfnet.graphml'

    taf = [0.001, 0.0001, 0.0005, 0.00001]
    traffic_matrix = gen_traffic_matrix(14, taf)
    adj_matrix, G = gen_lightpath_topo(path, wavelengths=4, draw=False)

    for src in range(len(traffic_matrix)):
        for dst in range(len(traffic_matrix[0])):
            service = traffic_matrix[src][dst]
            data_res = service[0]
            t = service[1]
            if data_res:
                print(nx.shortest_path(G, src, dst))




if __name__ == '__main__':
    heuristic_algorithm()

