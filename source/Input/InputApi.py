import numpy as np
import networkx as nx


class Input:
    def __init__(self):
        pass

    def read_graphml(self, path=None):
        "Read graphml data into memory."
        return nx.classes.graph.Graph

    def generate_topology(self, G=None, path=None):
        "Generate a lightpath topology based on graphml data."
        return nx.classes.multigraph.MultiGraph

    def generate_adjacency_martix(self, G):
        "Convert the topology to an adjacency matrix."
        return np.ndarray

    def generate_traffic_matrix(self, nodes: int):
        "Generate one traffic matrix."
        return np.ndarray
