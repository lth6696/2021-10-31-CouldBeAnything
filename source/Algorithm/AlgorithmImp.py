import math

import pandas as pd
import pulp.pulp
import networkx as nx
from pulp import *
from collections import defaultdict

from source.Algorithm.AlgorithmApi import Algorithm


class IntegerLinearProgram(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)

    def run(self, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix):
        prob = LpProblem("ServiceMapping", LpMaximize)
        solver = getSolver('CPLEX_CMD', timeLimit=10)

        row, col = adj_matrix.shape
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

        # bandwidth
        for i in nodes:
            for j in nodes:
                for t in range(adj_matrix[i][j]):
                    prob += (
                        lpSum([Lamda[s][d][k][i][j][t] * traffic_matrix[s][d][k].bandwidth for s in nodes for d in nodes for k in range(len(traffic_matrix[s][d]))])
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
                                    traffic_matrix[s][d][k].security / level_matrix[i][j][t] >= Lamda[s][d][k][i][j][t]
                                )
                                # extra condition to limit the level-cross, which can be ignore
                                prob += (
                                    (traffic_matrix[s][d][k].security / level_matrix[i][j][t] - 1 / 1e4) * Lamda[s][d][k][i][j][t] <= 1
                                )

        # The problem is solved using PuLP's choice of Solver
        prob.solve(solver=solver)

        NSuc = 0
        NTaf = 0
        NPut = 0
        for s in nodes:
            for d in nodes:
                for k, var in enumerate(S[s][d]):
                    if var.value() == 1.0:
                        NSuc += 1
                        NPut += traffic_matrix[s][d][k].bandwidth
                NTaf += len(traffic_matrix[s][d])

        logging.info("Status:{}".format(LpStatus[prob.status]))
        logging.info('IntegerLinearProgram - run - The successfully allocate bandwidth is {} Gbps.'.format(NPut))
        logging.info('IntegerLinearProgram - run - The number of request is {}.'.format(NTaf))
        logging.info('IntegerLinearProgram - run - We successfully map {}.'.format(NSuc))
        logging.info('IntegerLinearProgram - run - The mapping rate is {:2f}'.format(NSuc / NTaf * 100))

        return prob


class Heuristic(Algorithm):
    def __init__(self):
        super(Heuristic, self).__init__()

    def run(self, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, interval=3, step=0, multi_level=True):
        success_traffic = defaultdict(list)
        blocked_traffic = defaultdict(list)
        STANDARD_BANDWIDTH = 100

        # comb traffic, save traffic to the corresponding dict[level]
        convert_traffic_matrix, colums = self._convert_traffic(traffic_matrix, interval)
        for c in range(colums+len(convert_traffic_matrix.keys())*step):
            traffics = {}
            for i, key in enumerate(sorted(convert_traffic_matrix.keys())):
                if c-i*step < 0 or c-i*step >= colums:
                    continue
                else:
                    traffics[key] = convert_traffic_matrix[key][c-i*step]
            # traffics = {key: convert_traffic_matrix[key][c] for key in sorted(convert_traffic_matrix.keys())}
            G = nx.DiGraph()
            G.add_nodes_from([vertex for vertex in range(len(adj_matrix))])
            for level in sorted(list(traffics.keys())):
                if not multi_level:
                    G = nx.DiGraph()
                    G.add_nodes_from([vertex for vertex in range(len(adj_matrix))])
                # add edges into graph G
                for start in range(len(adj_matrix)):
                    for end in range(len(adj_matrix[start])):
                        self._update_lightpath(G, (start, end), level, STANDARD_BANDWIDTH, adj_matrix, level_matrix, bandwidth_matrix, multi_level=multi_level)

                # map traffic
                traffics[level].sort(key=lambda x: x[2])
                for traffic in traffics[level]:
                    src = traffic[0]
                    dst = traffic[1]
                    bandwidth = traffic[2]
                    try:
                        path = nx.shortest_path(G, src, dst)
                    except:
                        blocked_traffic[level].append(traffic)
                        continue
                    lightpaths = list(zip(path[:-1], path[1:]))

                    # 递归判断路径冗余并分配带宽
                    if self._map_service_to_lightpath(G, lightpaths, bandwidth, level, STANDARD_BANDWIDTH, adj_matrix, bandwidth_matrix, level_matrix, multi_level):
                        success_traffic[level].append(traffic)
                    else:
                        blocked_traffic[level].append(traffic)
        # todo 安全指标
        # todo 优化方向
        # for key in blocked_traffic:
        #     print(key)
        #     print(blocked_traffic[key])
        #     print('///////////////////////////////////////')

        Nsuc = sum([len(success_traffic[i]) for i in success_traffic])
        Nblo = sum([len(blocked_traffic[i]) for i in blocked_traffic])
        total = Nsuc + Nblo
        print('Success mapping rate is {:.2f}% in total {} traffic.'.format(Nsuc / total * 100, total))

    def _map_service_to_lightpath(self,
                                  G: nx.DiGraph,
                                  lightpaths: list,
                                  bandwidth: int,
                                  level: int,
                                  standard_bandwidth: int,
                                  adj_matrix: list,
                                  bandwidth_matrix: list,
                                  level_matrix: list,
                                  multi_level: bool):
        if not lightpaths:
            return False
        (start, end) = lightpaths[0]
        if standard_bandwidth / G[start][end]['cost'] < bandwidth:
            self._update_lightpath(G, (start, end), level, standard_bandwidth, adj_matrix, level_matrix, bandwidth_matrix, multi_level=multi_level)
            if standard_bandwidth / G[start][end]['cost'] < bandwidth:
                return False
        if lightpaths[1:] and not self._map_service_to_lightpath(G, lightpaths[1:], bandwidth, level, standard_bandwidth, adj_matrix, bandwidth_matrix, level_matrix, multi_level):
            return False
        else:
            index = G[start][end]['index']
            bandwidth_matrix[start][end][index] -= bandwidth
            G[start][end]['cost'] = standard_bandwidth / max(bandwidth_matrix[start][end][index], 1e-5)
            return True

    def _update_lightpath(self,
                          G: nx.DiGraph,
                          lightpath: tuple,
                          level: int,
                          standard_bandwidth: int,
                          adj_matrix: list,
                          level_matrix: list,
                          bandwidth_matrix: list,
                          multi_level: bool):
        start = lightpath[0]
        end = lightpath[1]
        if not adj_matrix[start][end]:
            return False
        # 找到剩余带宽最大的光路
        if multi_level:
            backup_lightpaths = [(t, bandwidth_matrix[start][end][t]) for t in range(adj_matrix[start][end]) if
                                 level_matrix[start][end][t] <= level]
        else:
            backup_lightpaths = [(t, bandwidth_matrix[start][end][t]) for t in range(adj_matrix[start][end]) if
                                 level_matrix[start][end][t] == level]
        if not backup_lightpaths:
            return False
        backup_lightpath = sorted(backup_lightpaths, key=lambda x: x[1], reverse=True)[0]
        backup_t = backup_lightpath[0]
        backup_bandwidth = backup_lightpath[1]
        # 若图G不存在连接关系，则添加连接
        if lightpath not in G.edges:
            G.add_edge(start, end)
        # 更新连接属性
        G[start][end]['level'] = level_matrix[start][end][backup_t]
        G[start][end]['index'] = backup_t
        G[start][end]['cost'] = standard_bandwidth / max(backup_bandwidth, 1e-5)
        return True

    def _convert_traffic(self, traffic_matrix: list, step: int = 3):
        traffic_temp = defaultdict(list)
        min_ = 1e10
        max_ = 0
        for src in range(len(traffic_matrix)):
            for dst in range(len(traffic_matrix[src])):
                for traffic in traffic_matrix[src][dst]:
                    if min_ > traffic.bandwidth:
                        min_ = traffic.bandwidth
                    elif max_ < traffic.bandwidth:
                        max_ = traffic.bandwidth
                    traffic_temp[traffic.security].append((src, dst, traffic.bandwidth))

        colums = math.ceil((max_-min_+1)/step)
        traffics = {}
        for level in traffic_temp:
            list_temp = [[] for _ in range(colums)]
            for traffic in traffic_temp[level]:
                index = int((traffic[2] - min_) / step)
                list_temp[index].append(traffic)
            list_temp = [sorted(row, key=lambda x: x[2]) for row in list_temp]
            traffics[level] = list_temp

        return traffics, colums

    def __run_v1(self, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, multi_level):
        success_traffic = defaultdict(list)
        blocked_traffic = defaultdict(list)
        STANDARD_BANDWIDTH = 100

        # comb traffic, save traffic to the corresponding dict[level]
        traffics = defaultdict(list)
        for src in range(len(traffic_matrix)):
            for dst in range(len(traffic_matrix[src])):
                for traffic in traffic_matrix[src][dst]:
                    traffics[traffic.security].append((src, dst, traffic.bandwidth))

        G = nx.DiGraph()
        G.add_nodes_from([vertex for vertex in range(len(adj_matrix))])
        for level in sorted(list(traffics.keys())):
            # add edges into graph G
            for start in range(len(adj_matrix)):
                for end in range(len(adj_matrix[start])):
                    self._update_lightpath(G, (start, end), level, STANDARD_BANDWIDTH, adj_matrix, level_matrix,
                                           bandwidth_matrix, multi_level)

            # map traffic
            traffics[level].sort(key=lambda x: x[2])
            for traffic in traffics[level]:
                src = traffic[0]
                dst = traffic[1]
                bandwidth = traffic[2]
                try:
                    path = nx.shortest_path(G, src, dst)
                except:
                    blocked_traffic[level].append(traffic)
                    continue
                lightpaths = list(zip(path[:-1], path[1:]))

                # 递归判断路径冗余并分配带宽
                if self._map_service_to_lightpath(G, lightpaths, bandwidth, level, STANDARD_BANDWIDTH, adj_matrix,
                                                  bandwidth_matrix, level_matrix, multi_level):
                    success_traffic[level].append(traffic)
                else:
                    blocked_traffic[level].append(traffic)

        Nsuc = sum([len(success_traffic[i]) for i in success_traffic])
        Nblo = sum([len(blocked_traffic[i]) for i in blocked_traffic])
        total = Nsuc + Nblo
        print('Success mapping rate is {:.2f}% in total {} traffic.'.format(Nsuc / total * 100, total))


class SuitableLightpathFirst(Algorithm):
    def __init__(self):
        super(SuitableLightpathFirst, self).__init__()

        self.success_traffic = defaultdict(list)
        self.blocked_traffic = defaultdict(list)
        self.default = 100

    def simulate(self, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, slf=True, multi_level=True):
        for src in range(len(traffic_matrix)):
            for dst in range(len(traffic_matrix)):
                for traffic in traffic_matrix[src][dst]:
                    if self._map_service((src, dst, traffic), adj_matrix, level_matrix, bandwidth_matrix, slf):
                        self.success_traffic[traffic.security].append(traffic)
                    else:
                        self.blocked_traffic[traffic.security].append(traffic)

        Nsuc = sum([len(self.success_traffic[i]) for i in self.success_traffic])
        Nblo = sum([len(self.blocked_traffic[i]) for i in self.blocked_traffic])
        total = Nsuc + Nblo
        print('Success mapping rate is {:.2f}% in total {} traffic.'.format(Nsuc / total * 100, total))

    def _map_service(self, traffic, adj_matrix, level_matrix, bandwidth_matrix, slf=True):
        src = traffic[0]
        dst = traffic[1]
        traffic = traffic[2]
        graph = nx.DiGraph()
        graph.add_nodes_from([vertex for vertex in range(len(adj_matrix))])
        if slf:
            self._add_lightpaths_v1(graph, traffic.bandwidth, traffic.security, adj_matrix, level_matrix, bandwidth_matrix)
        else:
            self._add_lightpaths(graph, traffic.security, adj_matrix, level_matrix, bandwidth_matrix)
        try:
            path = nx.shortest_path(graph, src, dst, weight='cost')
        except:
            return False
        lightpaths = list(zip(path[:-1], path[1:]))
        return self._allocate_bandwidth(graph, lightpaths, traffic.bandwidth, adj_matrix, bandwidth_matrix)

    def _add_lightpaths(self, graph, level, adj_matrix, level_matrix, bandwidth_matrix):
        for ori in range(len(adj_matrix)):
            for sin in range(len(adj_matrix)):
                parallel_lightpaths = [(t,
                                        level_matrix[ori][sin][t],
                                        bandwidth_matrix[ori][sin][t])
                                       for t in range(adj_matrix[ori][sin]) if level_matrix[ori][sin][t] <= level]
                if not parallel_lightpaths:
                    continue
                parallel_lightpaths.sort(key=lambda x: x[2], reverse=True)      # 降序
                lightpath = parallel_lightpaths[0]
                graph.add_edge(ori, sin)
                graph[ori][sin]['index'] = lightpath[0]
                graph[ori][sin]['level'] = lightpath[1]
                graph[ori][sin]['bandwidth'] = lightpath[2]
                graph[ori][sin]['cost'] = self.default / max(lightpath[2], 1e-5)

    def _add_lightpaths_v1(self, graph, bandwidth, level, adj_matrix, level_matrix, bandwidth_matrix):
        for ori in range(len(adj_matrix)):
            for sin in range(len(adj_matrix)):
                parallel_lightpaths = [(t,
                                        level_matrix[ori][sin][t],
                                        bandwidth_matrix[ori][sin][t])
                                       for t in range(adj_matrix[ori][sin])
                                       if level_matrix[ori][sin][t] <= level and bandwidth_matrix[ori][sin][t] >= bandwidth]
                if not parallel_lightpaths:
                    continue
                # parallel_lightpaths.sort(key=lambda x: x[1], reverse=True)  # 降序
                stay_level_lightpath = sorted([lightpath for lightpath in parallel_lightpaths if lightpath[1] == level],
                                              key=lambda x: x[2],
                                              reverse=True)
                if not stay_level_lightpath:
                    beyond_level_lightpath = [lightpath for lightpath in parallel_lightpaths if lightpath[1] < level]
                    if not beyond_level_lightpath:
                        continue
                    HIndex = 0
                    lightpath = beyond_level_lightpath[0]
                    for lp in beyond_level_lightpath:
                        h = lp[2]/(level-lp[1])
                        if h > HIndex:
                            lightpath = lp
                            HIndex = h
                        elif h == HIndex:
                            # todo 如果H相等，暂时优先考虑大带宽
                            if lp[2] > lightpath[2]:
                                lightpath = lp
                            else:
                                pass
                        else:
                            continue
                else:
                    lightpath = stay_level_lightpath[0]
                graph.add_edge(ori, sin)
                graph[ori][sin]['index'] = lightpath[0]
                graph[ori][sin]['level'] = lightpath[1]
                graph[ori][sin]['bandwidth'] = lightpath[2]
                graph[ori][sin]['cost'] = self.default / max(lightpath[2], 1e-5)

    def _allocate_bandwidth(self,
                            G: nx.DiGraph,
                            lightpaths: list,
                            bandwidth: int,
                            adj_matrix: list,
                            bandwidth_matrix: list):
        if not lightpaths:
            return False
        (start, end) = lightpaths[0]
        if G[start][end]['bandwidth'] < bandwidth:
            return False
        if lightpaths[1:] and not self._allocate_bandwidth(G, lightpaths[1:], bandwidth, adj_matrix, bandwidth_matrix):
            return False
        else:
            index = G[start][end]['index']
            bandwidth_matrix[start][end][index] -= bandwidth
            G[start][end]['bandwidth'] -= bandwidth
            G[start][end]['cost'] = self.default / max(bandwidth_matrix[start][end][index], 1e-5)
            return True