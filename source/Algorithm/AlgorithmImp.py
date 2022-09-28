import logging
import collections

import numpy as np
import pulp.pulp
import networkx as nx
from pulp import *

from source.result.ResultAnalysis import Result
from source.input.InputImp import Traffic


class IntegerLinearProgram(object):
    def __init__(self):
        pass

    def run(self, MultiDiG, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, scheme):
        prob = LpProblem("ServiceMapping", LpMaximize)
        # print(listSolvers(onlyAvailable=True))

        row, col = len(adj_matrix), len(adj_matrix)
        nodes = [i for i in range(row)]

        # Variables
        Lamda = []  # list structure, L[src][dst][i][j][t]
        for s in nodes:
            tempS = []
            for d in nodes:
                tempD = []
                for k in range(len(traffic_matrix)):
                    tempK = []
                    for i in nodes:
                        tempI = []
                        for j in nodes:
                            tempJ = []
                            for t in range(adj_matrix[i][j]):
                                # print(s, d, k, i, j, t)
                                var = LpVariable('L_{}_{}_{}_{}_{}_{}'.format(s, d, k, i, j, t),
                                                 lowBound=0,
                                                 upBound=1,
                                                 cat=LpInteger)
                                tempJ.append(var)
                            tempI.append(tempJ)
                        tempK.append(tempI)
                    tempD.append(tempK)
                tempS.append(tempD)
            Lamda.append(tempS)

        S = []  # list structure, S[src][dst].
        for s in nodes:
            temp_src = []
            for d in nodes:
                tempD = []
                for k in range(len(traffic_matrix)):
                    tempD.append(LpVariable("Suc_{}_{}_{}".format(s, d, k),
                                            lowBound=0, upBound=1, cat=LpInteger))
                temp_src.append(tempD)
            S.append(temp_src)

        # Objective
        # The objective function is added to 'prob' first
        prob += (
            lpSum([S[s][d][k] for s in nodes for d in nodes for k in range(len(traffic_matrix))])
        )

        # Constraints
        # continuity
        for s in nodes:
            for d in nodes:
                for k in range(len(traffic_matrix)):
                    prob += (
                            lpSum([Lamda[s][d][k][s][j][t] for j in nodes for t in range(adj_matrix[s][j])])
                            == S[s][d][k]
                    )
                    prob += (
                            lpSum([Lamda[s][d][k][i][d][t] for i in nodes for t in range(adj_matrix[i][d])])
                            == S[s][d][k]
                    )
                    prob += (
                            lpSum([Lamda[s][d][k][i][s][t] for i in nodes for t in range(adj_matrix[i][s])])
                            == 0
                    )
                    prob += (
                            lpSum([Lamda[s][d][k][d][j][t] for j in nodes for t in range(adj_matrix[d][j])])
                            == 0
                    )

        for s in nodes:
            for d in nodes:
                for k in range(len(traffic_matrix)):
                    for m in nodes:
                        if m == s or m == d:
                            continue
                        prob += (
                                lpSum([Lamda[s][d][k][i][m][t] for i in nodes for t in range(adj_matrix[i][m])])
                                == lpSum([Lamda[s][d][k][m][j][t] for j in nodes for t in range(adj_matrix[m][j])])
                        )
                        prob += (
                            lpSum([Lamda[s][d][k][i][m][t] for i in nodes for t in range(adj_matrix[i][m])]) <= 1
                        )
                        prob += (
                            lpSum([Lamda[s][d][k][m][j][t] for j in nodes for t in range(adj_matrix[m][j])]) <= 1
                        )

        # bandwidth
        for i in nodes:
            for j in nodes:
                for t in range(adj_matrix[i][j]):
                    prob += (
                            lpSum([Lamda[s][d][k][i][j][t] * traffic_matrix[k][s][d].bandwidth for s in nodes for d in
                                   nodes for k in range(len(traffic_matrix)) if traffic_matrix[k][s][d]])
                            <= bandwidth_matrix[i][j][t]
                    )

        for s in nodes:
            for d in nodes:
                for k in range(len(traffic_matrix)):
                    if not traffic_matrix[k][s][d]:
                        continue
                    prob += (
                            lpSum([Lamda[s][d][k][s][j][t] for j in nodes for t in range(adj_matrix[s][j])])
                            <= traffic_matrix[k][s][d].bandwidth
                    )
                    prob += (
                            lpSum([Lamda[s][d][k][i][d][t] for i in nodes for t in range(adj_matrix[i][d])])
                            <= traffic_matrix[k][s][d].bandwidth
                    )

        # level
        for s in nodes:
            for d in nodes:
                for k in range(len(traffic_matrix)):
                    if not traffic_matrix[k][s][d]:
                        continue
                    for i in nodes:
                        for j in nodes:
                            for t in range(adj_matrix[i][j]):
                                prob += (
                                        traffic_matrix[k][s][d].security / level_matrix[i][j][t] >=
                                        Lamda[s][d][k][i][j][t]
                                )
                                # extra condition to limit the level-cross, which can be ignore
                                if scheme == 'LSMS':
                                    prob += (
                                            (traffic_matrix[k][s][d].security / level_matrix[i][j][t] - 1 / 1e4) *
                                            Lamda[s][d][k][i][j][t] <= 1
                                    )

        # The problem is solved using PuLP's choice of Solver
        prob.solve(CPLEX_CMD(msg=False, timelimit=100))

        result = self._result_converter(Lamda,
                                        S,
                                        MultiDiG,
                                        traffic_matrix)
        return result

    def _result_converter(self,
                          lambda_: list,
                          s_: list,
                          graph: nx.MultiDiGraph,
                          traffic_matrix: np.ndarray
                          ):
        def find_next_node(
                current: int,
                variable: list
        ):
            # 从相邻节点内找出下一跳节点
            next = [
                (int(neighbor), t)
                for neighbor in graph.neighbors(str(current))    # 遍历邻居节点
                for t, var in enumerate(variable[current][int(neighbor)])   # 遍历平行边缘
                if round(var.value()) == 1   # 判断边缘是否被采用。因var可能为趋近1的小数（如:1.00001或0.99998），故需取整后判断。
            ]
            if len(next) != 1:
                raise Warning
            return next.pop()

        (K, row, col) = traffic_matrix.shape
        for k in range(K):
            for u in range(row):
                for v in range(col):
                    # 判断业务是否被路由
                    if s_[u][v][k].value() == 0.0:
                        continue
                    else:
                        traffic_matrix[k][u][v].blocked = False
                    # 获取路由业务路径
                    src = u
                    path = [src]
                    while True:
                        current = path[-1]
                        try:
                            (next, index) = find_next_node(current, lambda_[u][v][k])
                        except:
                            traffic_matrix[k][u][v].blocked = True
                            break
                        # 预留带宽
                        graph[str(current)][str(next)][index]['bandwidth'] -= traffic_matrix[k][u][v].bandwidth
                        graph[str(current)][str(next)][index]['traffic'].append(traffic_matrix[k][u][v])
                        # 记录链路等级
                        traffic_matrix[k][u][v].path_level.append(graph[str(current)][str(next)][index]['level'])
                        path.append(next)
                        # 若下一跳节点为目的节点，推出循环
                        if next == v:
                            break
                    traffic_matrix[k][u][v].path = path
        result = Result(graph, traffic_matrix)
        return result


class SecurityAwareServiceMappingAlgorithm(object):
    def __init__(self):
        self.routed_traffic = None

        # 默认常量，不可更改
        self.STANDARD_BANDWIDTH = 100
        self.ROUTED = 1
        self.BLOCKED = -1
        # 指标包含: 1、跨级带宽 2、同等级占用带宽比例 3、跨越等级 4、可用带宽
        self.MetricsName = {'CrossLevelBandwidth', 'UsedBandwidth', 'CrossLevel', 'AvailableBandwidth'}

    def solve(self,
              graph: nx.MultiDiGraph,
              traffic_matrix: np.matrix,
              scheme: str,
              metrics_weight: tuple
              ):
        """
        本方法求解
        :param graph: MultiDiGraph, 有向多边图
        :param traffic_matrix: np.matrix, 流量矩阵，目前数据结构为列表，后续更改为矩阵
        :param scheme: str, 两种方案，越级和非越级
        :param metrics_weight: tuple, 添加边缘时，多指标权重，权重总和为1
        :return: 仿真结果
        """
        (page, row, col) = traffic_matrix.shape

        # 初始化矩阵变量记录业务路由情况, 1为成功，-1为阻塞，0为业务不存在
        self.routed_traffic = np.zeros(shape=(page, row, col))

        # 为每条业务路由
        for k in range(page):
            for u in range(row):
                for v in range(col):
                    if u == v:
                        continue
                    if self._is_routed(traffic_matrix[k][u][v], graph, scheme, metrics_weight):
                        self.routed_traffic[k][u][v] = self.ROUTED
                        traffic_matrix[k][u][v].blocked = False
                    else:
                        self.routed_traffic[k][u][v] = self.BLOCKED

        # 统计实验结果
        re = Result(
            graph=graph,
            traffic_matrix=traffic_matrix
        )
        return re

    def _is_routed(self,
                   traffic: Traffic,
                   graph: nx.MultiDiGraph,
                   scheme: str,
                   metrics_weight: tuple
                   ):
        # 初始化单边有向图
        G = nx.DiGraph()
        G.add_nodes_from(graph.nodes)

        # 筛选并添加边缘
        self._add_edges(G, graph, traffic.security, scheme, metrics_weight)

        try:
            # 最短路径计算
            path = nx.shortest_path(G, str(traffic.src), str(traffic.dst), weight='cost')
            path_segments = list(zip(path[:-1], path[1:]))
            # 预留带宽
            is_reserved = self._reserve_bandwidth(G, graph, path_segments, traffic)
            if not is_reserved:
                raise ValueError
            traffic.path = path
        except:
            return False
        else:
            return True

    def _add_edges(self,
                   G: nx.DiGraph,
                   graph: nx.MultiDiGraph,
                   req_security: int,
                   scheme: str,
                   metrics_weight: tuple
                   ):
        """
        光路添加算法分为多个步骤：
        1 - 变量初始化
        2 - 边路所有边缘
        3 - 添加边缘
        :param G: nx.DiGraph, 单边有向图，算法内用于路径计算的图
        :param graph: nx.MultiDiGraph, 多边有向图，仿真拓扑图
        :param req_security: int, 请求安全等级
        :param scheme: str, 越级方案
        :param metrics_weight: tuple, 添加边缘时，多指标权重
        :return: bool, 布尔值
        """
        if len(metrics_weight) != len(self.MetricsName):
            raise ValueError('Wrong number of metric weights.')
        nodes = list(map(str, graph.nodes))
        for u in nodes:
            for v in nodes:
                # 判断节点对(u, v)间是否存在链路
                if not graph.has_edge(u, v):
                    continue
                # 找出等级相同的链路
                edges_with_same_level = {t: graph[u][v][t]
                                         for t in graph[u][v]
                                         if graph[u][v][t]['level'] == req_security}
                # 若存在等级相同的链路，则选出带宽最大的链路
                if edges_with_same_level:
                    t = max(edges_with_same_level.keys(), key=lambda x: edges_with_same_level[x]['bandwidth'])
                    edge = edges_with_same_level[t]
                # 若不存在等级相同的链路，则根据不同方案做出不同操作
                else:
                    if scheme == 'LSMS':
                        continue
                    elif scheme == 'LBMS':
                        # 找出(u, v)间高等级链路
                        edges_beyond_req_level = collections.OrderedDict({t: graph[u][v][t]
                                                                          for t in graph[u][v]
                                                                          if graph[u][v][t]['level'] < req_security})
                        if not edges_beyond_req_level:
                            continue
                        # 初始化指标矩阵大小为 指标数*链路数
                        metrics = np.zeros(shape=(len(self.MetricsName), len(edges_beyond_req_level)))
                        metrics[0] = [edges_beyond_req_level[t]['bandwidth'] / (req_security - edges_beyond_req_level[t]['level']) for t in edges_beyond_req_level]
                        metrics[1] = [np.sum([traffic.bandwidth
                                              for traffic in edges_beyond_req_level[t]['traffic']
                                              if traffic.security == edges_beyond_req_level[t]['level']])
                                      / (self.STANDARD_BANDWIDTH - edges_beyond_req_level[t]['bandwidth'])
                                      for t in edges_beyond_req_level]
                        metrics[2] = [req_security - edges_beyond_req_level[t]['level'] for t in edges_beyond_req_level]
                        metrics[3] = [edges_beyond_req_level[t]['bandwidth'] for t in edges_beyond_req_level]
                        metrics[np.isnan(metrics)] = 0
                        # 归一化
                        min_ = metrics.min(axis=1).reshape(len(self.MetricsName), 1)
                        max_ = metrics.max(axis=1).reshape(len(self.MetricsName), 1)
                        metrics = (metrics - min_) / (max_ - min_)
                        metrics[np.isnan(metrics)] = 0
                        metrics[2] = 1-metrics[2]
                        # 求最大指标对应的链路
                        max_metric_index = np.argmax(np.array(metrics_weight).dot(metrics))
                        t = list(edges_beyond_req_level.keys())[max_metric_index]
                        edge = edges_beyond_req_level[t]
                    else:
                        raise ValueError('Do not understand scheme: {}.'.format(scheme))
                G.add_edge(
                    u, v,
                    index=t,
                    level=edge['level'],
                    bandwidth=edge['bandwidth'],
                    cost=1-edge['bandwidth']/self.STANDARD_BANDWIDTH
                )
                # print(G[u][v]['cost'])

    def _reserve_bandwidth(self,
                           G: nx.DiGraph,
                           graph: nx.MultiDiGraph,
                           path_seg: list,
                           traffic: Traffic,
                           ):
        """
        本方法从前向后检查带宽多寡，从后向前预留带宽
        :param graph: nx.MultiDiGraph, 多边有向图
        :param path_seg: list, 路径分段集合，其包含多组(u, v)对
        :param traffic: obj, 流量对象
        :return: bool, 是否成功预留带宽
        """
        (u, v) = path_seg.pop(0)
        # 判断路径带宽是否足够
        if G[u][v]['bandwidth'] < traffic.bandwidth:
            return False
        # 若当前为最后一跳，即路径分段为空集合
        if not path_seg:
            graph[u][v][G[u][v]['index']]['bandwidth'] -= traffic.bandwidth
            graph[u][v][G[u][v]['index']]['traffic'].append(traffic)
            traffic.path_level.insert(0, G[u][v]['level'])
            return True
        # 若不为最后一跳，则向后递归
        else:
            # 若后继成功预留带宽，则继续向前预留
            if self._reserve_bandwidth(G, graph, path_seg, traffic):
                graph[u][v][G[u][v]['index']]['bandwidth'] -= traffic.bandwidth
                graph[u][v][G[u][v]['index']]['traffic'].append(traffic)
                traffic.path_level.insert(0, G[u][v]['level'])
                return True
            # 若后继带宽不足，则不会预留带宽
            else:
                return False

