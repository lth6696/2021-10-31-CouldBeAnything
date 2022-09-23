import logging
import math

import networkx
import numpy as np
import pandas as pd
import pulp.pulp
import networkx as nx
from pulp import *
from collections import defaultdict

from source.result.ResultAnalysis import Result
from source.input.InputImp import Traffic


class IntegerLinearProgram(object):
    def __init__(self):
        pass

    def run(self, MultiDiG, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, multi_level=True):
        prob = LpProblem("ServiceMapping", LpMaximize)
        solver = getSolver('CPLEX_CMD', timeLimit=100)

        row, col = len(adj_matrix), len(adj_matrix)
        nodes = [i for i in range(row)]

        # Variables
        Lamda = []  # list structure, L[src][dst][i][j][t]
        for s in nodes:
            tempS = []
            for d in nodes:
                tempD = []
                for k in range(len(traffic_matrix[s][d])):
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
                for k in range(len(traffic_matrix[s][d])):
                    tempD.append(LpVariable("Suc_{}_{}_{}".format(s, d, k),
                                            lowBound=0, upBound=1, cat=LpInteger))
                temp_src.append(tempD)
            S.append(temp_src)

        # Objective
        # The objective function is added to 'prob' first
        prob += (
            lpSum([S[s][d][k] for s in nodes for d in nodes for k in range(len(traffic_matrix[s][d]))])
        )

        # Constraints
        # continuity
        for s in nodes:
            for d in nodes:
                for k in range(len(traffic_matrix[s][d])):
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
                for k in range(len(traffic_matrix[s][d])):
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
                            lpSum([Lamda[s][d][k][i][j][t] * traffic_matrix[s][d][k].bandwidth for s in nodes for d in
                                   nodes for k in range(len(traffic_matrix[s][d]))])
                            <= bandwidth_matrix[i][j][t]
                    )

        for s in nodes:
            for d in nodes:
                for k in range(len(traffic_matrix[s][d])):
                    prob += (
                            lpSum([Lamda[s][d][k][s][j][t] for j in nodes for t in range(adj_matrix[s][j])])
                            <= traffic_matrix[s][d][k].bandwidth
                    )
                    prob += (
                            lpSum([Lamda[s][d][k][i][d][t] for i in nodes for t in range(adj_matrix[i][d])])
                            <= traffic_matrix[s][d][k].bandwidth
                    )

        # level
        for s in nodes:
            for d in nodes:
                for k in range(len(traffic_matrix[s][d])):
                    for i in nodes:
                        for j in nodes:
                            for t in range(adj_matrix[i][j]):
                                prob += (
                                        traffic_matrix[s][d][k].security / level_matrix[i][j][t] >=
                                        Lamda[s][d][k][i][j][t]
                                )
                                # extra condition to limit the level-cross, which can be ignore
                                if not multi_level:
                                    prob += (
                                            (traffic_matrix[s][d][k].security / level_matrix[i][j][t] - 1 / 1e4) *
                                            Lamda[s][d][k][i][j][t] <= 1
                                    )

        # The problem is solved using PuLP's choice of Solver
        prob.solve(solver=solver)

        NTaf = 0
        NPut = defaultdict(int)
        for s in nodes:
            for d in nodes:
                for k, var in enumerate(S[s][d]):
                    if var.value() == 1.0:
                        NPut[traffic_matrix[s][d][k].security] += traffic_matrix[s][d][k].bandwidth
                NTaf += len(traffic_matrix[s][d])

        logging.info('---------------------------------------------------------')
        logging.info("Status:{}".format(LpStatus[prob.status]))
        logging.info('IntegerLinearProgram - run - The successfully allocate bandwidth is {} Gbps.'.format(NPut))
        logging.info('IntegerLinearProgram - run - The number of request is {}.'.format(NTaf))

        blocked_traffic, success_traffic = self._result_converter(MultiDiG, Lamda, S, traffic_matrix)
        Nsuc = sum([len(success_traffic[i]) for i in success_traffic])
        Nblo = sum([len(blocked_traffic[i]) for i in blocked_traffic])
        logging.info('IntegerLinearProgram - run - We successfully map {}.'.format(Nsuc))
        logging.info('IntegerLinearProgram - run - The mapping rate is {:2f}'.format(Nsuc / NTaf * 100))
        result = Result()

        self.export_ilp_result(Lamda, S)
        # for v in prob.variables():
        #     logging.info("{}={}".format(v.name, v.varValue))
        return result

    def export_ilp_result(self, *args):
        print(len(args))
        return None

    def _result_converter(self, MultiDiG, Lambda, S, traffic_matrix):
        def callback():
            # logging.info("{} - End points are \n{}.".format(__name__, pd.DataFrame(edges[start])))
            end_point = {}
            for end in edges[start].keys():
                for t in edges[start][end].keys():
                    if Lambda[int(s)][int(d)][k][int(start)][int(end)][t].value() == 1.0:
                        if end in path:
                            continue
                        end_point[str(0+len(end_point.keys()))] = {'node': end, 'index': t}
            if len(end_point.keys()) == 0:
                logging.error('{} - No point has been found.'.format(__name__))
            else:
                if len(end_point.keys()) >= 2:
                    logging.warning("{} - There exist more than two end points {}!!".format(__name__, [(end_point[key]['node'], end_point[key]['index']) for key in end_point.keys()]))
                # logging.info("{} - Traffic origins from {} to {} employing {}-{}-{} as an intermediate.".
                #              format(__name__, s, d, start, end_point['0']['node'], end_point['0']['index']))
                path.append(end_point['0']['node'])
                lightpath[start] = {'sin': end_point['0']['node'],
                                    'index': end_point['0']['index'],
                                    'level': MultiDiG[start][end_point['0']['node']][end_point['0']['index']]['level']}
                MultiDiG[start][end_point['0']['node']][end_point['0']['index']]['bandwidth'] -= traffic_matrix[int(s)][int(d)][k].bandwidth
                return end_point['0']['node']

        success_traffic = defaultdict(list)
        blocked_traffic = defaultdict(list)
        edges = nx.to_dict_of_dicts(MultiDiG)
        for s in MultiDiG.nodes:
            for d in MultiDiG.nodes:
                for k, var in enumerate(S[int(s)][int(d)]):
                    traffic = traffic_matrix[int(s)][int(d)][k]
                    if var.value() == 0.0:
                        blocked_traffic[traffic.security].append(traffic)
                        # logging.info("{} - Traffic {} is blocked.".format(__name__, traffic.__dict__))
                        # logging.info("{}".format('-' * 100))
                        continue
                    traffic_matrix[int(s)][int(d)][k].blocked = False
                    # logging.info("{} - Traffic {} is mapped.".format(__name__, traffic.__dict__))
                    path = [s]
                    lightpath = {}
                    start = s

                    while True:
                        end = callback()
                        if end is None:
                            break
                        if end == d:
                            break
                        start = end
                    traffic_matrix[int(s)][int(d)][k].path = path
                    traffic_matrix[int(s)][int(d)][k].lightpath = lightpath
                    success_traffic[traffic.security].append(traffic)
                    logging.info("{} - The whole path is {}.".format(__name__, path))
                    logging.info("{}".format('-'*100))
        return blocked_traffic, success_traffic


class SecurityAwareServiceMappingAlgorithm(object):
    def __init__(self):
        self.routed_traffic = None

        # 默认常量，不可更改
        self.STANDARD_BANDWIDTH = 100
        self.ROUTED = 1
        self.BLOCKED = -1

    def solve(self,
              graph: nx.MultiDiGraph,
              traffic_matrix: np.matrix,
              scheme: str
              ):
        """
        本方法求解
        :param graph: MultiDiGraph, 有向多边图
        :param traffic_matrix: np.matrix, 流量矩阵，目前数据结构为列表，后续更改为矩阵
        :param str: str, 两种方案，越级和非越级
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
                    if self._is_routed(traffic_matrix[k][u][v], graph, scheme):
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
                   scheme: str
                   ):
        # 初始化单边有向图
        G = nx.DiGraph()
        G.add_nodes_from(graph.nodes)

        # 筛选并添加边缘
        self._add_edges(G, graph, traffic.security, scheme)

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
                   scheme: str
                   ):
        """
        光路添加算法分为多个步骤：
        1 - 变量初始化
        2 - 边路所有边缘
        3 - 添加边缘
        :param G: nx.DiGraph, 有向图
        :param req_security: int, 请求安全等级
        :param scheme: str, 越级方案
        :return: bool, 布尔值
        """
        nodes = list(map(str, graph.nodes))
        for u in nodes:
            for v in nodes:
                if not graph.has_edge(u, v):
                    continue
                if scheme == 'LBMS':
                    pass
                elif scheme == 'LSMS':
                    edges_with_same_level = {t: graph[u][v][t] for t in graph[u][v] if graph[u][v][t]['level'] == req_security}
                    if not edges_with_same_level:
                        continue
                    t = max(edges_with_same_level.keys(), key=lambda x: edges_with_same_level[x]['bandwidth'])
                    edge = edges_with_same_level[t]
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
            traffic.path_level.insert(0, G[u][v]['level'])
            return True
        # 若不为最后一跳，则向后递归
        else:
            # 若后继成功预留带宽，则继续向前预留
            if self._reserve_bandwidth(G, graph, path_seg, traffic):
                graph[u][v][G[u][v]['index']]['bandwidth'] -= traffic.bandwidth
                traffic.path_level.insert(0, G[u][v]['level'])
                return True
            # 若后继带宽不足，则不会预留带宽
            else:
                return False