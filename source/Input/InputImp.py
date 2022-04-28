import os
import logging
import numpy as np

import networkx as nx
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


class Traffic(object):
    DefaultValue = -1

    def __init__(self, **kwargs):
        self.src = Traffic.DefaultValue
        self.dst = Traffic.DefaultValue
        self.bandwidth = Traffic.DefaultValue
        self.security = Traffic.DefaultValue
        self.blocked = True

        self._set_value(**kwargs)

    def _set_value(self, **kwargs):
        input_keys = set(kwargs.keys())
        self_keys = set(self.__dict__.keys())

        for key in input_keys:
            if key not in self_keys:
                continue
            setattr(self, key, kwargs[key])

    def __setitem__(self, key, value):
        self_keys = set(self.__dict__.keys())
        if key in self_keys:
            self.__dict__[key] = value

    def __getitem__(self, item):
        return self.__dict__[item]


class InputImp(object):
    TopologyPath = "../graphml/nsfnet/nsfnet.graphml"

    def __init__(self):
        self.MultiDiG = nx.MultiDiGraph()
        self.req_bandwidth = 0

    def _get_customize_topology(self, path):
        if not os.path.exists(path):
            logging.error("Graphml file does not exist.")
            raise Exception("Invalid path {}!".format(path))
        G = nx.Graph(nx.read_graphml(path))
        logging.info("Successfully read the graphml file.")
        return G

    def _add_parallel_edge(self, origin: str, sink: str, index: int, level: int, bandwidth: int):
        self.MultiDiG.add_edge(origin, sink)
        # self.MultiDiG[origin][sink][index]['index'] = index
        self.MultiDiG[origin][sink][index]['level'] = level
        self.MultiDiG[origin][sink][index]['bandwidth'] = bandwidth

    def set_vertex_connection(self, path: str = '', nw: int = 4, nl: int = 3, bandwidth: int = 100):
        """
        :param path: string, the path to the custom topology.
        :param nw: int, the maximum number of lightpath (wavelength).
        :param nl: int, the number of divided levels.
        :param bandwidth: int, default initialized bandwidth.
        :return:
        """
        np.random.seed(0)
        if not path:
            path = InputImp.TopologyPath
        G = self._get_customize_topology(path)
        # 此处使用有向多径图表示节点间连接关系
        self.MultiDiG.add_nodes_from(G.nodes)
        for (origin, sink) in G.edges:
            # 随机生成并行光路数量，取值范围[1,nw+1)
            num_from_origin_to_sink = np.random.randint(1, nw+1)
            for i in range(num_from_origin_to_sink):
                self._add_parallel_edge(origin=origin,
                                        sink=sink,
                                        index=i,
                                        level=np.random.randint(1, nl+1),
                                        bandwidth=bandwidth)
            # G是无向图，故还需额外添加反向并行光路
            num_from_sink_to_origin = np.random.randint(1, nw+1)
            for i in range(num_from_sink_to_origin):
                self._add_parallel_edge(origin=sink,
                                        sink=origin,
                                        index=i,
                                        level=np.random.randint(1, nl+1),
                                        bandwidth=bandwidth)
        logging.info('Successfully generate the topology.')

    def get_adjacency_martix(self):
        try:
            adj_matrix = nx.adjacency_matrix(self.MultiDiG).todense()
            logging.info('Successfully generate the adjacency matrix.')
            return adj_matrix.tolist()       # return: list
        except:
            return []

    def get_level_matrix(self):
        nodes = list(self.MultiDiG.nodes)
        level_matrix = [[[] for _ in range(len(nodes))] for _ in range(len(nodes))]
        for (source, sink, index) in reversed(list(self.MultiDiG.edges)):
            level_matrix[int(source)][int(sink)].append(self.MultiDiG[source][sink][index]['level'])
        return level_matrix

    def get_bandwidth_matrix(self):
        nodes = list(self.MultiDiG.nodes)
        bandwidth_matrix = [[[] for _ in range(len(nodes))] for _ in range(len(nodes))]
        for (source, sink, index) in reversed(list(self.MultiDiG.edges)):
            bandwidth_matrix[int(source)][int(sink)].append(self.MultiDiG[source][sink][index]['bandwidth'])
        return bandwidth_matrix

    def get_traffic_matrix(self, nl: int = 3, nconn: int = 4, lam: int = 6):
        nodes = list(map(int, self.MultiDiG.nodes))
        traffic_matrix = [[[] for _ in nodes] for _ in nodes]

        np.random.seed(8)
        for r in nodes:
            for c in nodes:
                if r == c:
                    continue
                for _ in range(nconn):
                    bandwidth = np.random.poisson(lam)
                    traffic_matrix[r][c].append(
                        Traffic(src=r,
                                dst=c,
                                bandwidth=bandwidth,
                                security=np.random.randint(1, nl+1))
                    )
                    self.req_bandwidth += bandwidth
        logging.info('InputImp - generate_traffic_matrix - The total throughput is {} Gbps.'.format(self.req_bandwidth))
        return traffic_matrix


if __name__ == '__main__':
    t = InputImp()
    t.set_vertex_connection()
    adj = t.get_adjacency_martix()
    level = t.get_level_matrix()
    traffic_matrix = t.get_traffic_matrix()
    print(t.req_bandwidth)