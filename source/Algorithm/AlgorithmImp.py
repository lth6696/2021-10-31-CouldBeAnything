import logging
import math

import networkx
import pandas as pd
import pulp.pulp
import networkx as nx
from pulp import *
from collections import defaultdict

from source.Result.ResultAnalysis import Result
from source.Input.InputImp import Traffic
from source.simulator import LightPathBandwidth


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
        result.set_attrs(Nsuc, Nblo, NTaf, blocked_traffic, success_traffic, traffic_matrix, MultiDiG)

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


class SuitableLightpathFirst(object):
    def __init__(self):
        self.success_traffic = defaultdict(list)
        self.blocked_traffic = defaultdict(list)
        self.default = LightPathBandwidth

    def simulate(self, MultiDiG: nx.classes.multidigraph.MultiDiGraph, traffic_matrix: list, slf=True,
                 multi_level=True):
        ntraffic = 0
        for row in traffic_matrix:
            for col in row:
                for traffic in col:
                    ntraffic += 1
                    if self._map_service(traffic, MultiDiG, slf, multi_level):
                        self.success_traffic[traffic.security].append(traffic)
                        traffic.blocked = False
                    else:
                        self.blocked_traffic[traffic.security].append(traffic)
        Nsuc = sum([len(self.success_traffic[i]) for i in self.success_traffic])
        Nblo = sum([len(self.blocked_traffic[i]) for i in self.blocked_traffic])
        result = Result()
        result.set_attrs(Nsuc, Nblo, ntraffic, self.blocked_traffic, self.success_traffic, traffic_matrix, MultiDiG)
        return result

    def _map_service(self, traffic, MultiDiG, slf, multi_level):
        src = traffic.src
        dst = traffic.dst
        G = nx.DiGraph()
        G.add_nodes_from(MultiDiG.nodes)
        if slf:
            self._add_lightpaths_slf(G, traffic, MultiDiG)
        else:
            self._add_lightpaths_ff(G, traffic, MultiDiG, multi_level)
        try:
            path = nx.shortest_path(G, str(src), str(dst), weight='cost')
        except:
            traffic.block_reason = '0x01'
            return False
        lightpaths = list(zip(path[:-1], path[1:]))
        if self._allocate_bandwidth(G, MultiDiG, lightpaths, traffic):
            traffic.path = path
            return True
        else:
            traffic.block_reason = '0x02'
            return False

    def _add_lightpaths_ff(self, graph, traffic, MultiDiG, multi_level):
        nodes = list(MultiDiG.nodes)
        for ori in nodes:
            for sin in nodes:
                try:
                    parallel_lightpaths = dict(MultiDiG[ori][sin])
                    lightpath = (-1, 0)  # (index, ava_bandwidth)
                    for index in parallel_lightpaths:
                        if parallel_lightpaths[index]['level'] > traffic.security:
                            continue
                        if multi_level == False and parallel_lightpaths[index]['level'] < traffic.security:
                            continue
                        # 基于First-Fit思路，最大可用带宽优先
                        bandwidth_to_compare = parallel_lightpaths[index]['bandwidth']
                        if bandwidth_to_compare > lightpath[1]:
                            lightpath = (index, bandwidth_to_compare)
                        else:
                            continue
                    if lightpath == (-1, 0):
                        raise Exception('Something went wrong.')
                    graph.add_edge(ori, sin)
                    graph[ori][sin]['index'] = lightpath[0]
                    graph[ori][sin]['level'] = parallel_lightpaths[lightpath[0]]['level']
                    graph[ori][sin]['bandwidth'] = lightpath[1]
                    graph[ori][sin]['cost'] = self.default / max(float(lightpath[1]), 1e-5)
                except:
                    # 不存在并行光路
                    continue

    def _add_lightpaths_slf(self, graph, traffic, MultiDiG):
        nodes = list(MultiDiG.nodes)
        for ori in nodes:
            for sin in nodes:
                try:
                    parallel_lightpaths = dict(MultiDiG[ori][sin])
                    lightpath_equal_level = sorted([(index, parallel_lightpaths[index]['bandwidth'])
                                                    for index in parallel_lightpaths
                                                    if parallel_lightpaths[index]['level'] == traffic.security],
                                                   key=lambda x: x[1],
                                                   reverse=True)
                    # 若不存在需求等级光路
                    if not lightpath_equal_level:
                        # 找出高于需求的光路
                        lightpath_bigger_level = sorted(
                            [(index, parallel_lightpaths[index]['bandwidth'] / (
                                    traffic.security - parallel_lightpaths[index]['level']))
                             for index in parallel_lightpaths
                             if parallel_lightpaths[index]['level'] < traffic.security],
                            key=lambda x: x[1],
                            reverse=True
                        )
                        # 若光路不存在，则抛出异常
                        if not lightpath_bigger_level:
                            raise Exception('Something went wrong.')
                        symbol_ligthpath = lightpath_bigger_level[0]
                        lightpath_equal_hindex = [lightpath for lightpath in lightpath_bigger_level
                                                  if lightpath[1] == symbol_ligthpath[1]]
                        # 若存在多个相等权重的光路
                        if lightpath_equal_hindex:
                            # 比较光路非本等级业务抢占的带宽，取小值
                            lightpath_occupy_bandwidth = []
                            for lightpath in lightpath_equal_hindex:
                                occupy_traffic = MultiDiG[ori][sin][lightpath[0]]['traffic']
                                if occupy_traffic:
                                    occupy_bandwidth = sum([t[2] for t in occupy_traffic if t[3] != traffic.security])
                                    lightpath_occupy_bandwidth.append((lightpath[0], occupy_bandwidth))
                                else:
                                    lightpath_occupy_bandwidth.append((lightpath[0], 0))
                            lightpath_occupy_bandwidth.sort(key=lambda x: x[1], reverse=False)
                            lightpath = lightpath_occupy_bandwidth[0]
                        else:
                            # 去H最大的高等光路
                            lightpath = lightpath_bigger_level[0]
                    else:
                        # 取带宽最大的同等级光路
                        lightpath = lightpath_equal_level[0]
                    graph.add_edge(ori, sin)
                    index = lightpath[0]
                    bandwidth = MultiDiG[ori][sin][lightpath[0]]['bandwidth']
                    graph[ori][sin]['index'] = index
                    graph[ori][sin]['level'] = MultiDiG[ori][sin][index]['level']
                    graph[ori][sin]['bandwidth'] = bandwidth
                    graph[ori][sin]['cost'] = self.default / max(bandwidth, 1e-5)
                except:
                    continue

    def _allocate_bandwidth(self,
                            G: nx.DiGraph,
                            MultiDiG: nx.MultiDiGraph,
                            lightpaths: list,
                            traffic: Traffic,
                            ):
        if not lightpaths:
            return False
        (start, end) = lightpaths[0]
        if G[start][end]['bandwidth'] < traffic.bandwidth:
            return False
        if lightpaths[1:] and not self._allocate_bandwidth(G, MultiDiG, lightpaths[1:], traffic):
            return False
        else:
            index = G[start][end]['index']
            MultiDiG[start][end][index]['bandwidth'] -= traffic.bandwidth
            MultiDiG[start][end][index]['traffic'].append(
                (traffic.src, traffic.dst, traffic.bandwidth, traffic.security))
            traffic.lightpath[start] = {'sin': end, 'index': index, 'level': G[start][end]['level']}
            return True


class MetricDrivenPathSelect(object):
    def __init__(self):
        self.success_traffic = defaultdict(list)
        self.blocked_traffic = defaultdict(list)
        self.default = LightPathBandwidth

    def simulate(self,
                 MultiDiG: nx.classes.multidigraph.MultiDiGraph,
                 traffic_matrix: list,
                 metric: str,
                 weight: tuple,
                 multi_level: bool = True,
                 cutoff: int = 5):
        # 'MBF-HOP' is a benchmark when nl = 1
        metrics = ['LLF', 'MBF']
        if metric not in metrics:
            raise ValueError('Invalid metric \'{}\''.format(metric))
        ntraffic = 0
        for row in range(len(traffic_matrix)):
            for col in range(len(traffic_matrix[row])):
                for traffic in traffic_matrix[row][col]:
                    ntraffic += 1
                    if self._map_service(traffic, MultiDiG, multi_level, cutoff, metric, weight):
                        self.success_traffic[traffic.security].append(traffic)
                        traffic.blocked = False
                    else:
                        self.blocked_traffic[traffic.security].append(traffic)

        Nsuc = sum([len(self.success_traffic[i]) for i in self.success_traffic])
        Nblo = sum([len(self.blocked_traffic[i]) for i in self.blocked_traffic])
        result = Result()
        result.set_attrs(Nsuc, Nblo, ntraffic, self.blocked_traffic, self.success_traffic, traffic_matrix, MultiDiG)
        return result

    def _map_service(self, traffic, MultiDiG, multi_level, cutoff, metric, weight):
        src = traffic.src
        dst = traffic.dst
        G = nx.DiGraph()
        G.add_nodes_from(MultiDiG.nodes)
        # assign lightpath
        self._add_lightpaths(G, traffic, MultiDiG, multi_level, metric)
        # KSP algorithm and first-fit algorithm
        try:
            paths = list(nx.all_simple_paths(G, str(src), str(dst), cutoff))
        except:
            traffic.block_reason = '0x01'
            return False
        if not paths:
            traffic.block_reason = '0x01'
            return False
        # path select
        path = self._select_path(G, traffic, paths, weight)
        # service map
        lightpaths = list(zip(path[:-1], path[1:]))
        if self._allocate_bandwidth(G, MultiDiG, lightpaths, traffic):
            traffic.path = lightpaths
            return True
        else:
            traffic.block_reason = '0x02'
            return False

    def _select_path(self,
                     G: nx.DiGraph,
                     traffic: Traffic,
                     paths: list,
                     weight: tuple
                     ):
        """
        Four metric for path selection:
        1 - level deviation
        2 - average available bandwidth
        3 - hop
        """
        metric_value = {'level': [], 'bandwidth': [], 'hop': [], 'total': []}
        if len(weight) != len(metric_value.keys())-1:
            raise ValueError('Invalid weight \'{}\''.format(weight))
        for p in paths:
            path = list(zip(p[:-1], p[1:]))
            metric_value['level'].append((sum(
                [(G[start][end]['level'] - traffic.security) ** 2 for (start, end) in path]
            ) / len(path)) ** 0.5)
            metric_value['bandwidth'].append(sum([G[start][end]['bandwidth'] for (start, end) in path]) / len(path))
            metric_value['hop'].append(len(path))
        metric_value['total'] = [
            weight[0] * metric_value['level'][i] / max(max(metric_value['level']), 1e-7) +
            weight[1] * (1 - metric_value['bandwidth'][i] / max(max(metric_value['bandwidth']), 1e-7)) +
            weight[2] * metric_value['hop'][i] / max(metric_value['hop']) for i in range(len(paths))
        ]
        # select a path with the minimum metric value
        selected = metric_value['total'].index(min(metric_value['total']))
        return paths[selected]

    def _add_lightpaths(self,
                        graph: nx.DiGraph,
                        traffic: Traffic,
                        MultiDiG: nx.MultiDiGraph,
                        multi_level: bool,
                        metric: str):
        metrics = {'LLF': 'level', 'MBF': 'bandwidth'}
        if metric not in metrics.keys():
            raise ValueError("Invalid metric \'{}\'".format(metric))
        nodes = list(MultiDiG.nodes)
        for ori in nodes:
            for sin in nodes:
                try:
                    parallel_lightpaths = dict(MultiDiG[ori][sin])
                    lightpath = (-1, 0)  # (index, level/ava_bandwidth)
                    for index in parallel_lightpaths:
                        if parallel_lightpaths[index]['level'] > traffic.security:
                            continue
                        if multi_level == False and parallel_lightpaths[index]['level'] < traffic.security:
                            continue
                        if parallel_lightpaths[index]['bandwidth'] < traffic.bandwidth:
                            continue
                        # 基于First-Fit思路，最低等级优先 or 最大可用带宽优先 【总是选择最高值】
                        to_compare = parallel_lightpaths[index][metrics[metric]]
                        if to_compare > lightpath[1]:
                            lightpath = (index, to_compare)
                        else:
                            continue
                    if lightpath == (-1, 0):
                        raise Exception('Something went wrong.')
                    graph.add_edge(ori, sin)
                    graph[ori][sin]['index'] = lightpath[0]
                    graph[ori][sin]['level'] = parallel_lightpaths[lightpath[0]]['level']
                    graph[ori][sin]['bandwidth'] = parallel_lightpaths[lightpath[0]]['bandwidth']
                    graph[ori][sin]['cost'] = self.default / max(float(parallel_lightpaths[lightpath[0]]['bandwidth']), 1e-5)
                except:
                    # 不存在并行光路
                    continue

    def _allocate_bandwidth(self,
                            G: nx.DiGraph,
                            MultiDiG: nx.MultiDiGraph,
                            lightpaths: list,
                            traffic: Traffic,
                            ):
        if not lightpaths:
            return False
        (start, end) = lightpaths[0]
        if G[start][end]['bandwidth'] < traffic.bandwidth:
            return False
        if lightpaths[1:] and not self._allocate_bandwidth(G, MultiDiG, lightpaths[1:], traffic):
            return False
        else:
            index = G[start][end]['index']
            MultiDiG[start][end][index]['bandwidth'] -= traffic.bandwidth
            MultiDiG[start][end][index]['traffic'].append(
                (traffic.src, traffic.dst, traffic.bandwidth, traffic.security))
            traffic.lightpath[start] = {'sin': end, 'index': index, 'level': G[start][end]['level']}
            return True


class ShortestPathFirst(object):
    def __init__(self):
        self.success_traffic = defaultdict(list)
        self.blocked_traffic = defaultdict(list)
        self.default = LightPathBandwidth

    def simulate(self,
                 MultiDiG: nx.classes.multidigraph.MultiDiGraph,
                 traffic_matrix: list,
                 multi_level: bool = True,
                 ):
        ntraffic = 0
        for row in range(len(traffic_matrix)):
            for col in range(len(traffic_matrix[row])):
                for traffic in traffic_matrix[row][col]:
                    ntraffic += 1
                    if self._map_service(traffic, MultiDiG, multi_level):
                        self.success_traffic[traffic.security].append(traffic)
                        traffic.blocked = False
                    else:
                        self.blocked_traffic[traffic.security].append(traffic)

        Nsuc = sum([len(self.success_traffic[i]) for i in self.success_traffic])
        Nblo = sum([len(self.blocked_traffic[i]) for i in self.blocked_traffic])
        result = Result()
        result.set_attrs(Nsuc, Nblo, ntraffic, self.blocked_traffic, self.success_traffic, traffic_matrix, MultiDiG)
        return result

    def _map_service(self, traffic, MultiDiG, multi_level):
        src = traffic.src
        dst = traffic.dst
        G = nx.DiGraph()
        G.add_nodes_from(MultiDiG.nodes)
        # assign lightpath
        self._add_lightpaths(G, traffic, MultiDiG, multi_level)
        # KSP algorithm and first-fit algorithm
        try:
            path = nx.shortest_path(G, str(src), str(dst), weight='cost')
        except:
            traffic.block_reason = '0x01'
            return False
        if not path:
            traffic.block_reason = '0x01'
            return False
        # path select
        # service map
        lightpaths = list(zip(path[:-1], path[1:]))
        if self._allocate_bandwidth(G, MultiDiG, lightpaths, traffic):
            traffic.path = lightpaths
            return True
        else:
            traffic.block_reason = '0x02'
            return False

    def _add_lightpaths(self,
                        graph: nx.DiGraph,
                        traffic: Traffic,
                        MultiDiG: nx.MultiDiGraph,
                        multi_level: bool):
        nodes = list(MultiDiG.nodes)
        for ori in nodes:
            for sin in nodes:
                try:
                    parallel_lightpaths = dict(MultiDiG[ori][sin])
                    lightpath = (-1, 0)  # (index, level/ava_bandwidth)
                    for index in parallel_lightpaths:
                        if parallel_lightpaths[index]['bandwidth'] < traffic.bandwidth:
                            continue
                        # 基于First-Fit思路，最低等级优先 or 最大可用带宽优先 【总是选择最高值】
                        to_compare = parallel_lightpaths[index]['bandwidth']
                        if to_compare > lightpath[1]:
                            lightpath = (index, to_compare)
                        else:
                            continue
                    if lightpath == (-1, 0):
                        raise Exception('Something went wrong.')
                    graph.add_edge(ori, sin)
                    graph[ori][sin]['index'] = lightpath[0]
                    graph[ori][sin]['level'] = parallel_lightpaths[lightpath[0]]['level']
                    graph[ori][sin]['bandwidth'] = parallel_lightpaths[lightpath[0]]['bandwidth']
                    graph[ori][sin]['cost'] = self.default / max(float(parallel_lightpaths[lightpath[0]]['bandwidth']), 1e-5)
                except:
                    # 不存在并行光路
                    continue

    def _allocate_bandwidth(self,
                            G: nx.DiGraph,
                            MultiDiG: nx.MultiDiGraph,
                            lightpaths: list,
                            traffic: Traffic,
                            ):
        if not lightpaths:
            return False
        (start, end) = lightpaths[0]
        if G[start][end]['bandwidth'] < traffic.bandwidth:
            return False
        if lightpaths[1:] and not self._allocate_bandwidth(G, MultiDiG, lightpaths[1:], traffic):
            return False
        else:
            index = G[start][end]['index']
            MultiDiG[start][end][index]['bandwidth'] -= traffic.bandwidth
            MultiDiG[start][end][index]['traffic'].append(
                (traffic.src, traffic.dst, traffic.bandwidth, traffic.security))
            traffic.lightpath[start] = {'sin': end, 'index': index, 'level': G[start][end]['level']}
            return True
