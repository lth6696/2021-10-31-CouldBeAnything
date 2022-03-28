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

    def generate_traffic_matrix(self, nodes: list, nconn: int = 0, nbandwdith: float = 0.0):
        """
        Generate one traffic matrix.
        nconn: The number of connections between source and sink nodes.
        If "nconn" is default value, a random number connections will be generated.
        Minimum bandwidth requirements for end-user services:
        +---------+-----------+----------+
        | Service | Bandwidth | Security |
        +---------+-----------+----------+
        |   VoIP  |     5     |     1    |
        +---------+-----------+----------+
        |   SSH   |    1.5    |     1    |
        +---------+-----------+----------+
        |  Video  |     4     |     2    |
        +---------+-----------+----------+
        |   IPTV  |    12.5   |     2    |
        +---------+-----------+----------+
        |  Cloud  |    18     |     2    |
        +---------+-----------+----------+
        |  Live   |    12.5   |     3    |
        +---------+-----------+----------+
        |  Gaming |     1.5   |     3    |
        +---------+-----------+----------+
        """
        return list
