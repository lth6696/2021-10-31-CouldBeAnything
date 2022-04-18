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

    def run(self, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix):
        success_traffic = []
        blocked_traffic = []
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
                    self._update_lightpath(G, (start, end), level, STANDARD_BANDWIDTH, adj_matrix, level_matrix, bandwidth_matrix)

            # map traffic
            traffics[level].sort(key=lambda x: x[2])
            for traffic in traffics[level]:
                src = traffic[0]
                dst = traffic[1]
                bandwidth = traffic[2]
                try:
                    path = nx.shortest_path(G, src, dst)
                except:
                    blocked_traffic.append(traffic)
                    continue
                lightpaths = list(zip(path[:-1], path[1:]))

                # 递归判断路径冗余并分配带宽
                if self._map_service_to_lightpath(G, lightpaths, bandwidth, level, STANDARD_BANDWIDTH, adj_matrix, bandwidth_matrix, level_matrix):
                    success_traffic.append(traffic)
                else:
                    blocked_traffic.append(traffic)

        print(len(success_traffic) + len(blocked_traffic))
        print(len(success_traffic) / (len(success_traffic) + len(blocked_traffic)) * 100)

    def _map_service_to_lightpath(self,
                                  G: nx.DiGraph,
                                  lightpaths: list,
                                  bandwidth: int,
                                  level: int,
                                  standard_bandwidth: int,
                                  adj_matrix: list,
                                  bandwidth_matrix: list,
                                  level_matrix: list):
        if not lightpaths:
            return False
        (start, end) = lightpaths[0]
        if standard_bandwidth / G[start][end]['cost'] < bandwidth:
            self._update_lightpath(G, (start, end), level, standard_bandwidth, adj_matrix, level_matrix, bandwidth_matrix)
            if standard_bandwidth / G[start][end]['cost'] < bandwidth:
                return False
        if lightpaths[1:] and not self._map_service_to_lightpath(G, lightpaths[1:], bandwidth, level, standard_bandwidth, adj_matrix, bandwidth_matrix, level_matrix):
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
                          bandwidth_matrix: list):
        start = lightpath[0]
        end = lightpath[1]
        if not adj_matrix[start][end]:
            return False
        # 找到剩余带宽最大的光路
        backup_lightpaths = [(t, bandwidth_matrix[start][end][t]) for t in range(adj_matrix[start][end]) if
                             level_matrix[start][end][t] <= level]
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
