"""
The RWKA model for the PuLP modeller

Authors: LTH6696 2021
"""
import pandas as pd
import random
import logging
import logging.config

import pulp.pulp
from pulp import *

pd.set_option('display.max_column', 200)
pd.set_option('display.width', 200)

def rwka_function():
    prob = LpProblem("BD", LpMaximize)

    solver = getSolver('CPLEX_CMD', timeLimit=10)

    # Creates a list of all nodes
    nodes = [i for i in range(6)]
    levels = [0.1, 0.3, 0.5, 0.8]

    # Creates adjust matrix
    adj_matrix, adj_weight = generate_adj_matrix(len(nodes), levels)

    # Creates traffic matrix
    traffic_matrix = generate_traffic_matrix_v2(len(nodes), levels)

    # Inputs
    MaxLightpathRes = [[40 * min(j, 1) for j in i] for i in adj_matrix]

    # -------------------------------------------- Variables -------------------------------------------------------- #
    # Lamda indicates whether the key resource is successfully allocated.
    Lamda = []  # list structure, K[src][dst][i][j][t]
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_dst = []
            for i in nodes:
                temp_i = []
                for j in nodes:
                    temp_j = []
                    if adj_matrix[i][j]:
                        for t in range(adj_matrix[i][j]):
                            temp_j.append(LpVariable("ResAlloc_{}_{}_{}_{}_{}".format(s, d, i, j, t),
                                                     lowBound=0,
                                                     upBound=1,
                                                     cat=LpInteger))
                    else:
                        temp_j.append(0)
                    temp_i.append(temp_j)
                temp_dst.append(temp_i)
            temp_src.append(temp_dst)
        Lamda.append(temp_src)

    # S indicates whether the request for data res is successfully allocated.
    S = []  # list structure, S[src][dst].
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_src.append(LpVariable("SucAlloc_{}_{}".format(s, d),
                                       lowBound=0, upBound=1, cat=LpInteger))
        S.append(temp_src)

    # -------------------------------------------- Objective -------------------------------------------------------- #
    # The objective function is added to 'prob' first
    prob += (
        lpSum([S[s][d] for s in nodes for d in nodes])
    )

    # ------------------------------------------- Constraints ------------------------------------------------------- #
    # continuity constraint
    for s in nodes:
        for d in nodes:
            prob += (
                lpSum([
                    lpSum([Lamda[s][d][s][j][t] for t in range(adj_matrix[s][j])]) for j in nodes if adj_matrix[s][j]
                ]) == S[s][d]
            )
            prob += (
                lpSum([
                    lpSum([Lamda[s][d][i][d][t] for t in range(adj_matrix[i][d])]) for i in nodes if adj_matrix[i][d]
                ]) == S[s][d]
            )
            for k in nodes:
                if k != s and k != d:
                    prob += (
                        lpSum([
                            lpSum([Lamda[s][d][i][k][t] for t in range(adj_matrix[i][k])]) for i in nodes if adj_matrix[i][k]
                        ]) ==
                        lpSum([
                            lpSum([Lamda[s][d][k][j][t] for t in range(adj_matrix[k][j])]) for j in nodes if adj_matrix[k][j]
                        ])
                    )
            prob += (
                lpSum([
                    lpSum([Lamda[s][d][i][s][t] for t in range(adj_matrix[i][s])]) for i in nodes if adj_matrix[i][s]
                ]) == 0
            )
            prob += (
                lpSum([
                    lpSum([Lamda[s][d][d][j][t] for t in range(adj_matrix[d][j])]) for j in nodes if adj_matrix[d][j]
                ]) == 0
            )

    # maximum resource constraint
    for i in nodes:
        for j in nodes:
            if adj_matrix[i][j]:
                for t in range(adj_matrix[i][j]):
                    prob += (
                        lpSum([Lamda[s][d][i][j][t] * traffic_matrix[s][d][0] for s in nodes for d in nodes]) <=
                        MaxLightpathRes[i][j] * 1 / (adj_weight[i][j][t] + 1)
                    )
                    prob += (
                        lpSum([Lamda[s][d][i][j][t] * traffic_matrix[s][d][0] * traffic_matrix[s][d][1] for s in nodes for d in nodes]) <=
                        MaxLightpathRes[i][j] * adj_weight[i][j][t] / (adj_weight[i][j][t] + 1)
                    )

    for s in nodes:
        for d in nodes:
            prob += (
                lpSum([
                    lpSum([Lamda[s][d][s][j][t] for t in range(adj_matrix[s][j])]) for j in nodes
                ]) <= traffic_matrix[s][d][0]
            )
            prob += (
                lpSum([
                    lpSum([Lamda[s][d][i][d][t] for t in range(adj_matrix[i][d])]) for i in nodes
                ]) <= traffic_matrix[s][d][0]
            )

    # level constraints
    for s in nodes:
        for d in nodes:
            for i in nodes:
                for j in nodes:
                    for t in range(adj_matrix[i][j]):
                        if traffic_matrix[s][d][1] > 0:
                            prob += (
                                traffic_matrix[s][d][1] / adj_weight[i][j][t] >= Lamda[s][d][i][j][t]
                            )

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # print("Status:", LpStatus[prob.status])
    logging.info("Status:{}".format(LpStatus[prob.status]))
    # for v in prob.variables():
    #     logging.info("{} = {}".format(v.name, v.varValue))
    logging.info("Adj Matrix: \n{}".format(pd.DataFrame(adj_matrix)))
    logging.info("Traffic Matrix: \n{}".format(pd.DataFrame(traffic_matrix)))
    logging.info("Res Alloc Situation: \n{}".format(output_matrix(S)))
    statistic_lightpath_utilization(Lamda, traffic_matrix, adj_matrix)
    for s in nodes:
        for d in nodes:
            if S[s][d].value():
                logging.info("{}-{} : \n{}".format(s, d, output_matrix_v2(Lamda[s][d])))
    logging.info("Not satisfy the level constraint: {}".format(check_level_constraint(S, Lamda, traffic_matrix, adj_weight)))
    logging.info("Total throughput is :{}, alloc is :{}".format(sum([sum([j[0] for j in i]) for i in traffic_matrix]),
                                                                sum([sum([traffic_matrix[i][j][0] * S[i][j].value() for j in nodes]) for i in nodes])))


def output_matrix(matrix):
    M = []
    for i in matrix:
        m = []
        for j in i:
            if type(j) == pulp.LpVariable:
                if j.value() is None:
                    m.append(0)
                    continue
                m.append(int(j.value()))
            else:
                m.append(0)
        M.append(m)
    return pd.DataFrame(M)


def output_matrix_v2(matrix):
    M = []
    for i in matrix:
        m = []
        for j in i:
            v = []
            for t in j:
                if type(t) == pulp.LpVariable:
                    if t.value() is None:
                        v.append(0)
                        continue
                    v.append(int(t.value()))
                else:
                    v.append(0)
            m.append(sum(v))
        M.append(m)
    return pd.DataFrame(M)

def statistic_lightpath_utilization(Lamda, traffic_matrix, adj_matrix):
    ligthpath_util_data = [[[0 for t in range(adj_matrix[i][j])] for j in range(len(adj_matrix))] for i in range(len(adj_matrix))]
    ligthpath_util_key = [[[0 for t in range(adj_matrix[i][j])] for j in range(len(adj_matrix))] for i in range(len(adj_matrix))]
    nodes = len(adj_matrix)
    for s in range(nodes):
        for d in range(nodes):
            for i in range(nodes):
                for j in range(nodes):
                    if adj_matrix[i][j]:
                        for t in range(adj_matrix[i][j]):
                            if Lamda[s][d][i][j][t].value() == 1:
                                data = traffic_matrix[s][d][0]
                                key = traffic_matrix[s][d][0] * traffic_matrix[s][d][1]
                                ligthpath_util_data[i][j][t] += data
                                ligthpath_util_key[i][j][t] += key
    logging.info("Light path data resource usage: \n{}".format(pd.DataFrame(ligthpath_util_data)))
    logging.info("Light path key resource usage: \n{}".format(pd.DataFrame(ligthpath_util_key)))


def generate_adj_matrix(nodes, levels):
    # row represents src node, col represents dst node.
    # adj_matrix = [[random.randint(0, 4) if i != j else 0 for j in range(nodes)] for i in range(nodes)]
    # adj_weight = [[[levels[random.randint(0,len(levels)-1)] for t in range(j)] for j in i] for i in adj_matrix]
    adj_matrix = [[0, 3, 0, 0, 1, 0],
                  [4, 0, 4, 0, 4, 0],
                  [0, 1, 0, 3, 0, 1],
                  [0, 0, 3, 0, 0, 4],
                  [1, 2, 0, 0, 0, 1],
                  [0, 0, 2, 1, 2, 0]]
    adj_weight = [[[],                      [0.1, 0.8, 0.1],    [],                     [],                 [0.1],                  []],
                  [[0.3, 0.3, 0.3, 0.8],    [],                 [0.8, 0.3, 0.3, 0.1],   [],                 [0.3, 0.1, 0.5, 0.1],   []],
                  [[],                      [0.3],              [],                     [0.5, 0.1, 0.3],    [],                     [0.1]],
                  [[],                      [],                 [0.1, 0.8, 0.3],        [],                 [],                     [0.3, 0.3, 0.1, 0.3]],
                  [[0.1],                   [0.3, 0.5],         [],                     [],                 [],                     [0.8]],
                  [[],                      [],                 [0.8, 0.1],             [0.1],              [0.1, 0.3],             []]]
    return adj_matrix, adj_weight


def generate_traffic_matrix_v2(nodes, levels):
    # The traffic matrix that records the requests (rate, level) from origin node to termination node.
    # traffic_matrix = [[(0, 0) for _ in range(nodes)] for _ in range(nodes)]
    # for s in range(nodes):
    #     for d in range(nodes):
    #         if s == d: continue
    #         rate = random.randint(0, 10)
    #         level = rate % len(levels)
    #         traffic_matrix[s][d] = (rate, levels[level])
    traffic_matrix = [[(0, 0), (6, 0.5), (7, 0.8), (2, 0.5), (1, 0.3), (8, 0.1)],
                      [(8, 0.1), (0, 0), (8, 0.1), (9, 0.3), (0, 0), (5, 0.3)],
                      [(0, 0), (8, 0.1), (0, 0), (5, 0.3), (2, 0.5), (7, 0.8)],
                      [(1, 0.3), (10, 0.5), (8, 0.1), (0, 0), (9, 0.3), (0, 0)],
                      [(8, 0.1), (4, 0.1), (4, 0.1), (1, 0.3), (0, 0), (9, 0.3)],
                      [(6, 0.5), (4, 0.1), (0, 0), (2, 0.5), (6, 0.5), (0, 0)]]
    return traffic_matrix

def check_level_constraint(S, Lamda, traffic_matrix, adj_weight):
    nodes = len(traffic_matrix)
    res = []
    for s in range(nodes):
        for d in range(nodes):
            if traffic_matrix[s][d][0] > 0 and S[s][d].value() > 0:
                for i in range(nodes):
                    for j in range(nodes):
                        for t in range(len(adj_weight[i][j])):
                            if Lamda[s][d][i][j][t].value() == 1:
                                if traffic_matrix[s][d][1] < adj_weight[i][j][t]:
                                    res.append((Lamda[s][d][i][j][t], traffic_matrix[s][d][1], adj_weight[i][j][t]))
    return res

if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')
    rwka_function()
