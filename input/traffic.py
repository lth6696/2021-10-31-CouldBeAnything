from configparser import ConfigParser
import logging

import networkx as nx
import numpy as np


class Traffic:
    """
    算力网络内分三类节点：边缘节点、路由节点、算力节点
    业务仅从边缘节点发起，去往算力节点
    业务属性包含：
        源节点
        宿节点
        带宽
        算力
        存储
    """

    def __init__(self, **kwargs):
        # 算力网络内业务的目的地需要由网管决策
        self.__name = 'traffic'
        self.src = -1
        self.req_bandwidth = 0  # Gb/s
        self.req_compute = 0
        self.req_storage = 0
        self.req_latency = 0
        self.data = 0           # GB
        self.assign_variables(**kwargs)

    def assign_variables(self, **kwargs):
        if not kwargs:
            return
        if not set(kwargs.keys()).issubset(set(self.__dict__.keys())):
            return
        for attr in kwargs:
            setattr(self, attr, kwargs[attr])
        return


class TrafficMatrix:
    """
    流量矩阵，矩阵行表示源节点，矩阵列表示宿节点。
    矩阵内每个元素包含至多一个Traffic对象。
    """

    def __init__(
            self,
            graph: nx.DiGraph,
            config_file: str = 'simulation_setting.ini'
    ):
        self.graph = graph
        self.client_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['type'] == 'client']

        # 业务平均请求带宽服从为lambda的泊松分布
        self.poisson_lambda_bandwidth = 0
        self.poisson_lambda_data = 0
        self.min_req_compute = 0
        self.max_req_compute = 0
        self.min_req_storage = 0
        self.max_req_storage = 0
        self.min_req_latency = 0
        self.max_req_latency = 0

        self.cf = config_file
        self._init_parameters()

    def generate_traffic_matrices(self, K: int, **kwargs):
        """
        本方法用于生成K张流量矩阵
        :param K: int, 流量矩阵张数
        :param kwargs: dict, 仅开始递归需要传入参数
        :return: 流量矩阵
        """
        if "tm" in kwargs.keys():
            traffic_matrices = kwargs['tm']
        else:
            traffic_matrices = []
        if K > 0:
            self._init_traffic_matrix(traffic_matrices)
            self.generate_traffic_matrices(K-1, tm=traffic_matrices)
        return traffic_matrices

    def _init_parameters(self):
        section_service = 'service'
        cfg = ConfigParser()
        is_read_success = cfg.read(self.cf)
        if not is_read_success:
            logging.error("{} - {} - Config file can not be read!".format(__file__, __name__))
            raise ValueError("Config file can not be read!")
        if not cfg.has_section(section_service):
            logging.error("{} - {} - The section of '{}' is not in {}".format(__file__, __name__, section_service, cfg.sections()))
            raise ValueError("The config file does not have '{}' section.".format(section_service))
        # 配置类内属性
        for attr in cfg[section_service]:
            if hasattr(self, attr):
                setattr(self, attr, int(cfg[section_service][attr]))
            else:
                logging.error("{} - {} - '{}' does not belong to the class.".format(__file__, __name__, attr))
        return True

    def _init_traffic_matrix(self, tm: list):
        """
        本方法用于生成单一流量矩阵
        :param tm: list, 存储流量矩阵的列表
        """
        traffic_matrix = np.empty(len(self.client_nodes), dtype=object)
        for i, node in enumerate(self.client_nodes):
            traffic_matrix[i] = Traffic(
                src=int(node),
                req_bandwidth=int(np.random.poisson(self.poisson_lambda_bandwidth))/100,     # Gb/s
                req_compute=np.random.randint(self.min_req_compute, self.max_req_compute),
                req_storage=np.random.randint(self.min_req_storage, self.max_req_storage),
                req_latency=np.random.randint(self.min_req_latency, self.max_req_latency),
                data=np.random.poisson(self.poisson_lambda_data)/10     # GB
            )
        tm.append(traffic_matrix)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.random.poisson(lam=1, size=100000)
    pillar = 15
    plt.hist(x, bins=50)
    plt.show()
