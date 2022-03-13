import random
import numpy as np
import networkx as nx
import logging
import pandas as pd


class Traffic(object):
    def __init__(self, src, dst, resource, t):
        self.src = src
        self.dst = dst
        self.resource = resource
        self.data = None
        self.key = None
        self.t = t
        self.path = []

        self.__cal_resource()

    def __cal_resource(self):
        self.data = self.resource * 1 / (self.t + 1)
        self.key = self.resource * self.t / (self.t + 1)


class LightPath(object):
    def __init__(self, start, end, index, resource, t):
        self.start = start
        self.end = end
        self.index = index
        self.resource = resource
        self.ava_data = 0
        self.ava_key = 0
        self.t = t

        self.__cal_ava_resource()

    def __cal_ava_resource(self):
        self.ava_data = self.resource * 1 / (self.t + 1)
        self.ava_key = self.resource * self.t / (self.t + 1)

    def max_data_resource(self):
        return self.resource * 1 / (self.t + 1)

    def max_key_resource(self):
        return self.resource * self.t / (self.t + 1)


# generate traffic matrix
def gen_traffic_matrix(nodes, tafs, max_res):
    random.seed(1)
    traffic_matrix = [[None for _ in range(nodes)] for _ in range(nodes)]
    for src in range(nodes):
        for dst in range(nodes):
            resource = np.abs(np.random.normal(loc=1, scale=0.4)) * max_res         # traffic load
            if src != dst and resource != 0:
                traffic_matrix[src][dst] = Traffic(src,
                                                   dst,
                                                   resource,
                                                   tafs[random.randint(0, len(tafs)-1)]
                                                   )
    return traffic_matrix


# generate lightpath topology
def gen_lightpath_topo(path, wavelengths, tafs, max_res):
    random.seed(1)
    G = nx.Graph(nx.read_graphml(path))
    adj_matrix = np.array(nx.adjacency_matrix(G).todense())
    adj_matrix_with_lightpath = [[[] for _ in range(len(G.nodes))] for _ in range(len(G.nodes))]
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix)):
            adj_matrix[row][col] = adj_matrix[row][col] * random.randint(1, wavelengths)
            for t in range(adj_matrix[row][col]):
                t_index = len(tafs) - 1 - t % len(tafs)
                # t_index = random.randint(0, len(tafs)-1)
                # logging.info("Lightpath {}-{}-{} level {}".format(row, col, t, tafs[t_index]))
                lightpath = LightPath(row, col, t, max_res, tafs[t_index])
                adj_matrix_with_lightpath[row][col].append(lightpath)
    logging.info("Adj Matrix\n{}".format(pd.DataFrame(adj_matrix)))
    return G, adj_matrix_with_lightpath
