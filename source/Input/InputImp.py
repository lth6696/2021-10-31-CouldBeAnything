import os
import logging
import networkx as nx
import numpy as np
import pandas
import random

from .InputApi import Input


class Service:
    def __init__(self, bandwidth, security):
        self.bandwidth = bandwidth
        self.security = security


class InputImp(Input):
    def __init__(self):
        Input.__init__(self)

    def read_graphml(self, path=None):
        if not path:
            path = '../topology/nsfnet/nsfnet.graphml'
        if not os.path.exists(path):
            logging.error("Graphml file does not exist.")
            raise Exception("Invalid path {}!".format(path))
        G = nx.Graph(nx.read_graphml(path))
        logging.info("Successfully read the graphml file.")
        return G

    def generate_topology(self, G=None, path=''):
        if not G and not path:
            raise Exception("An input is required.")
        if G:
            MG = nx.MultiGraph(G)
        elif os.path.exists(path) and not G:
            G = self.read_graphml(path)
            MG = nx.MultiGraph(G)
        else:
            logging.error("Can not generate topology.")
            raise Exception("An input is required.")
        logging.info('Successfully generate the topology.')
        return MG

    def generate_adjacency_martix(self, G):
        adj_matrix = np.array(nx.adjacency_matrix(G).todense())
        logging.info('Successfully generate the adjacency matrix.')
        return adj_matrix

    def generate_lightpath_adjacency_matrix(self, adj_matrix, nlp=4):
        if type(adj_matrix) != np.ndarray:
            logging.error('InputImp - generate_lightpath_adjacency_matrix - Got wrong type.')
            raise Exception('Input type error.')
        row, col = adj_matrix.shape
        for i in range(row):
            for j in range(col):
                adj_matrix[i][j] = adj_matrix[i][j] * random.randint(1, nlp)
        return adj_matrix

    def generate_lightpath_level_matrix(self, adj_matrix, levels: list):
        if not levels:
            logging.error('InputImp - generate_lightpath_level_matrix - Got invalid input.')
            raise Exception('Invalid input.')
        row, col = adj_matrix.shape
        level_matrix = [
            [
                [
                    levels[random.randint(0, len(levels) - 1)] for _ in range(adj_matrix[i][j])
                ] for j in range(col)
            ] for i in range(row)
        ]
        return level_matrix

    def generate_lightpath_bandwidth(self, adj_matrix, bandwidth):
        row, col = adj_matrix.shape
        bandwidth_matrix = [
            [
                [
                    bandwidth for _ in range(adj_matrix[i][j])
                ] for j in range(col)
            ] for i in range(row)
        ]
        return bandwidth_matrix

    def generate_traffic_matrix(self, nodes: list, nconn: int = 1, nbandwdith: float = 0.0):
        if not nodes:
            logging.error('InputImp - generate_traffic_matrix - args is empty.')
            raise Exception('Empty args')
        services = [Service(5, 1),
                    Service(1.5, 1),
                    Service(4, 0.5),
                    Service(12.5, 0.5),
                    Service(18, 0.5),
                    Service(12.5, 0.1),
                    Service(1.5, 0.1)]
        traffic_matrix = [[[] for _ in nodes] for _ in nodes]
        row = col = len(nodes)
        # print(pandas.DataFrame(traffic_matrix))

        for r in range(row):
            for c in range(col):
                for _ in range(nconn):
                    index = random.randint(0, len(services)-1)
                    traffic_matrix[r][c].append(services[index])
        # print(pandas.DataFrame(traffic_matrix))
        return traffic_matrix
