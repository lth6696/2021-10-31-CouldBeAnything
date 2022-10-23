import logging
from configparser import ConfigParser

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Topology:
    """
    算力网络由三类节点组成：边缘节点、路由节点、算力节点
    算力节点包含计算资源、存储资源
    边缘属性包含带宽
    """
    def __init__(self, config_file='simulation_setting.ini'):
        self.cfg = ConfigParser()
        self.cf = config_file
        self.__name = 'can_topology'
        self.graphml_file = ''

        # 节点属性
        self.num_router = 0
        self.num_datacenter = 0
        self.num_edge = 0
        self.num_client = 0

        # 链路属性，单位Gb/s
        self.bandwidth_client_to_edge = 0
        self.bandwidth_datacenter_to_router = 0
        self.bandwidth_edge_to_router = 0
        self.bandwidth_router_to_router = 0

        # 算力节点属性，计算资源单位GIPS，存储资源单位GB
        self.computing_edge = 0
        self.computing_datacenter = 0
        self.storage_edge = 0
        self.storage_datacenter = 0

        self._init_func()

    def generate_topology(self, draw: bool = False):
        """
        边缘连接规则：
        1、用户节点仅和路由节点相连
        2、算力节点仅和路由节点相连
        3、路由节点可以和任意其它两类节点相连
        :return:
        """
        G = nx.DiGraph(nx.read_graphml(self.graphml_file))
        for (u, v) in G.edges:
            # 配置链路带宽
            var_name = 'bandwidth_{}_to_{}'.format(*sorted([G.nodes[u]['type'], G.nodes[v]['type']]))
            G[u][v]['bandwidth'] = int(getattr(self, var_name))
            G[u][v]['max_bandwidth'] = int(getattr(self, var_name))
            G[u][v]['cost'] = 100 / G[u][v]['bandwidth']
        for node in G.nodes:
            G.nodes[node]['max_compute'] = G.nodes[node]['compute']
            G.nodes[node]['max_storage'] = G.nodes[node]['storage']
        if draw:
            nx.draw(
                G,
                with_labels=True,
                node_color=["#FFDF53" for _ in range(int(self.num_router))] +
                           ["#FF7739" for _ in range(int(self.num_datacenter))] +
                           ["#58842E" for _ in range(int(self.num_edge))] +
                           ["#C9EAD7" for _ in range(int(self.num_client))],
                node_size=[500 for _ in range(int(self.num_router))] +
                          [800 for _ in range(int(self.num_datacenter))] +
                          [500 for _ in range(int(self.num_edge))] +
                          [300 for _ in range(int(self.num_client))]
            )
            plt.show()
        return G

    def get_neighbors(self, G: nx.DiGraph):
        """
        本方法获取所有节点的邻居关系
        :param G: nx.DiGraph, 有向图
        :return: dict, 记录邻居关系的列表，格式{node1: [neighbor1, neighbor2,...]}。其中，节点类型为int。
        """
        # 若图无节点
        if not G.nodes:
            return {}
        neighbors = {int(node): [int(neighbor) for neighbor in G.neighbors(node)] for node in G.nodes}
        return neighbors

    def get_bandwidth_matrix(self, G: nx.DiGraph):
        """
        本方法获取边缘带宽矩阵
        :param G: nx.DiGraph, 有向图
        :return: np.array, 带宽矩阵
        """
        bandwidth_matrix = np.zeros((len(G.nodes), len(G.nodes)))
        for (u, v) in G.edges:
            bandwidth_matrix[int(u)][int(v)] = G[u][v]['bandwidth']
        return np.array(bandwidth_matrix)

    def get_compute_and_storage_matrix(self, G: nx.DiGraph):
        """
        本方法获取计算资源矩阵
        :param G: nx.DiGraph, 有向图
        :return: np.array, 带宽矩阵
        """
        compute_matrix = np.zeros(len(G.nodes), dtype=float)
        storage_matrix = np.zeros(len(G.nodes), dtype=float)
        for node in G.nodes:
            if G.nodes[node]['type'] == 'datacenter':
                compute_matrix[int(node)] = int(G.nodes[node]['compute'])
                storage_matrix[int(node)] = int(G.nodes[node]['storage'])
        return compute_matrix, storage_matrix

    def _init_func(self):
        is_read_success = self.cfg.read(self.cf)
        if not is_read_success:
            logging.error("{} - {} - Config file can not be read!".format(__file__, __name__))
            raise ValueError("Config file can not be read!")
        section = 'network'
        if not self.cfg.has_section(section):
            logging.error("{} - {} - The section of '{}' is not in {}".format(__file__, __name__, section, self.cfg.sections()))
            raise ValueError("The config file does not have '{}' section.".format(section))
        # 配置类内属性
        for attr in self.cfg[section]:
            if hasattr(self, attr):
                setattr(self, attr, self.cfg[section][attr])
            else:
                logging.error("{} - {} - '{}' does not belong to the class.".format(__file__, __name__, attr))
        return True