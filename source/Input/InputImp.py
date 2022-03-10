import os
import logging
import networkx as nx
import numpy as np

from .InputApi import Input


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

    def generate_traffic_matrix(self, nodes: int):
        return None
