import networkx as nx
import numpy as np


class Result(object):
    """
    统计单次实验结果，结果包含如下几种：
    1 - 成功率 t
    2 - 平均跳数 t
    3 - 吞吐量 t
    4 - 平均链路利用率 g
    5 - 安全等级偏差 t
    """
    def __init__(self,
                 graph: nx.MultiDiGraph,
                 traffic_matrix: np.ndarray
                 ):
        # 默认值
        self.LightPathBandwidth = 100 # Gb/s
        self.InitialValue = -1

        # 输入
        self.graph = graph
        self.traffic_matrix = traffic_matrix

        # 结果变量
        self.mapping_rate = 0
        self.throughput = 0
        self.req_bandwidth = 0  # 平均每个节点请求总带宽的大小
        self.ave_hops = 0
        self.ave_link_utilization = 0
        self.ave_level_deviation = 0

        self._get_results()

    def _get_results(self):
        (page, row, col) = self.traffic_matrix.shape

        # 初始化空矩阵
        routed_traffic_matrix = np.zeros(shape=(page, row, col))
        throughput_matrix = np.zeros(shape=(page, row, col))
        req_bandwidth_matrix = np.zeros(shape=(row, col))
        ave_hops_matrix = np.ones(shape=(page, row, col)) * self.InitialValue
        ave_level_deviation_matrix = np.ones(shape=(page, row, col)) * self.InitialValue

        for k in range(page):
            for u in range(row):
                for v in range(col):
                    traffic = self.traffic_matrix[k][u][v]
                    if traffic is None:
                        continue
                    req_bandwidth_matrix[u][v] += traffic.bandwidth
                    if traffic.blocked == False:
                        routed_traffic_matrix[k][u][v] = 1
                        ave_hops_matrix[k][u][v] = len(traffic.path)-1
                        throughput_matrix[k][u][v] = traffic.bandwidth
                        ave_level_deviation_matrix[k][u][v] = (np.sum((traffic.security - np.array(traffic.path_level))**2)/len(traffic.path_level))**0.5

        ave_link_utilization_matrix = np.ones(shape=(row, col)) * self.InitialValue
        for (u, v, t) in self.graph.edges:
            link_utilization = (1 - self.graph[u][v][t]['bandwidth'] / self.LightPathBandwidth) * 100
            [u, v] = map(int, [u, v])
            if ave_link_utilization_matrix[u][v] == self.InitialValue:
                ave_link_utilization_matrix[u][v] = link_utilization
            else:
                ave_link_utilization_matrix[u][v] = np.mean([link_utilization, ave_link_utilization_matrix[u][v]])

        self.mapping_rate = np.sum(routed_traffic_matrix) / (page*row*(col-1))
        self.throughput = np.sum(throughput_matrix)  # Gb/s
        self.req_bandwidth = np.mean(req_bandwidth_matrix)
        self.ave_hops = np.mean(ave_hops_matrix[ave_hops_matrix != self.InitialValue])
        self.ave_link_utilization = np.mean(ave_link_utilization_matrix[ave_link_utilization_matrix != self.InitialValue])
        self.ave_level_deviation = np.mean(ave_level_deviation_matrix[ave_level_deviation_matrix != self.InitialValue])
