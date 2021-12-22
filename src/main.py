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


def rwka_function():
    prob = LpProblem("BD", LpMaximize)

    # Creates a list of all nodes
    nodes = [i for i in range(6)]

    # Creates adjust matrix
    adj_matrix = generate_adj_matrix(len(nodes))

    # Creates traffic matrix
    # traffic_matrix = generate_traffic_matrix(len(nodes), 10)
    # traffic_mat_num = cal_traffic_mat_req_num(traffic_matrix)
    traffic_matrix = generate_traffic_matrix_v2(len(nodes))

    # Inputs
    ReservedKeyBw = [[4*j for j in i] for i in adj_matrix]
    ReservedDataBw = [[8*j for j in i] for i in adj_matrix]

    # Variables
    # K indicates whether the key resource is successfully allocated.
    K = []  # list structure, K[src][dst][i][j]
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_dst = []
            for i in nodes:
                temp_i = []
                for j in nodes:
                    if adj_matrix[i][j]:
                        temp_i.append(LpVariable("KeyBwSucAlloc_{}_{}_{}_{}".format(s, d, i, j),
                                                 lowBound=0,
                                                 cat=LpInteger))
                    else:
                        temp_i.append(0)
                temp_dst.append(temp_i)
            temp_src.append(temp_dst)
        K.append(temp_src)

    # T indicates whether the data resource is successfully allocated.
    T = []  # list structure, T[src][dst][i][j]
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_dst = []
            for i in nodes:
                temp_i = []
                for j in nodes:
                    if adj_matrix[i][j]:
                        temp_i.append(LpVariable("DataBwSucAlloc_{}_{}_{}_{}".format(s, d, i, j),
                                                 lowBound=0,
                                                 cat=LpInteger))
                    else:
                        temp_i.append(0)
                temp_dst.append(temp_i)
            temp_src.append(temp_dst)
        T.append(temp_src)

    # ST indicates whether the request for data res is successfully allocated.
    ST = []  # list structure, S[src][dst].
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_src.append(LpVariable("DataSucAllo_{}_{}".format(s, d),
                                       lowBound=0, cat=LpInteger))
        ST.append(temp_src)

    # SK indicates whether the request for key res is successfully allocated.
    SK = []  # list structure, S[src][dst].
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_src.append(LpVariable("KeySucAllo_{}_{}".format(s, d),
                                       lowBound=0, cat=LpInteger))
        SK.append(temp_src)

    # The objective function is added to 'prob' first
    prob += (
        lpSum([ST[s][d] for s in nodes for d in nodes])
        # lpSum([ST[s][d]+SK[s][d] for s in nodes for d in nodes])
    )

    # The follow constraints are entered
    # Data Res
    # continuous constraint
    for s in nodes:
        for d in nodes:
            # Here we can limit the number of hops.
            prob += (
                lpSum([T[s][d][s][j] for j in nodes if adj_matrix[s][j]]) == ST[s][d]
            )
            prob += (
                lpSum([T[s][d][i][d] for i in nodes if adj_matrix[i][d]]) == ST[s][d]
            )
            for k in nodes:
                prob += (
                    lpSum([T[s][d][i][k] for i in nodes if adj_matrix[i][k]]) ==
                    lpSum([T[s][d][k][j] for j in nodes if adj_matrix[k][j]])
                )

    # maximum resource constraint
    for i in nodes:
        for j in nodes:
            if adj_matrix[i][j]:
                prob += (
                    lpSum([T[s][d][i][j] for s in nodes for d in nodes]) <= ReservedDataBw[i][j]
                )

    for s in nodes:
        for d in nodes:
            prob += (
                lpSum([T[s][d][s][j] for j in nodes if adj_matrix[s][j]]) <= traffic_matrix[s][d]
            )
            prob += (
                lpSum([T[s][d][i][d] for i in nodes if adj_matrix[i][d]]) <= traffic_matrix[s][d]
            )

    # # outgoing & incoming
    # for s in nodes:
    #     for d in nodes:
    #         prob += (
    #             lpSum([T[s][d][i][s] for i in nodes]) == 0
    #         )

    # Key Res
    # key & data constraint
    # for s in nodes:
    #     for d in nodes:
    #         for k in nodes:
    #             prob += (
    #                 lpSum([K[i][k] for i in nodes if adj_matrix[i][k]]) ==
    #                 lpSum([T[i][k] for i in nodes if adj_matrix[i][k]])
    #             )

    # # continuous constraint
    # for s in nodes:
    #     for d in nodes:
    #         # Here we can limit the number of hops.
    #         prob += (
    #             lpSum([K[s][d][s][j] for j in nodes if adj_matrix[s][j]]) == SK[s][d]
    #         )
    #         prob += (
    #             lpSum([K[s][d][i][d] for i in nodes if adj_matrix[i][d]]) == SK[s][d]
    #         )
    #         for k in nodes:
    #             prob += (
    #                 lpSum([K[s][d][i][k] for i in nodes if adj_matrix[i][k]]) ==
    #                 lpSum([K[s][d][k][j] for j in nodes if adj_matrix[k][j]])
    #             )
    #
    # # maximum resource constraint
    # for i in nodes:
    #     for j in nodes:
    #         if adj_matrix[i][j]:
    #             prob += (
    #                 lpSum([K[s][d][i][j] for s in nodes for d in nodes]) <= ReservedKeyBw[i][j]
    #             )
    #
    # for s in nodes:
    #     for d in nodes:
    #         prob += (
    #             lpSum([K[s][d][s][j] for j in nodes if adj_matrix[s][j]]) <= traffic_matrix[s][d]
    #         )
    #         prob += (
    #             lpSum([K[s][d][i][d] for i in nodes if adj_matrix[i][d]]) <= traffic_matrix[s][d]
    #         )

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # print("Status:", LpStatus[prob.status])
    logging.info("Status:{}".format(LpStatus[prob.status]))
    # for v in prob.variables():
    #     logging.info("{} = {}".format(v.name, v.varValue))
    logging.info("Adj Matrix: \n{}".format(pd.DataFrame(adj_matrix)))
    logging.info("Traffic Matrix: \n{}".format(pd.DataFrame(traffic_matrix)))
    logging.info("Data Alloc Situation: \n{}".format(output_matrix(ST)))
    for s in nodes:
        for d in nodes:
            if ST[s][d].value():
                logging.info("{}-{} Data Alloc Path: \n{}".format(s, d, output_matrix(T[s][d])))
    logging.info("Total throughput is :{}".format(sum([sum(i) for i in traffic_matrix])))


def output_matrix(matrix):
    M = []
    for i in matrix:
        m = []
        for j in i:
            if type(j) == pulp.LpVariable:
                m.append(int(j.value()))
            else:
                m.append(0)
        M.append(m)
    return pd.DataFrame(M)


def generate_adj_matrix(nodes):
    # row represents src node, col represents dst node.
    # adj_matrix = [[random.randint(0, 1) if i != j else 0 for j in range(nodes)] for i in range(nodes)]
    adj_matrix = [[0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1],
                  [0, 0, 1, 1, 1, 0]]
    return adj_matrix


def generate_traffic_matrix(nodes, num_req):
    traffic_matrix = [[[] for _ in range(nodes)] for _ in range(nodes)]
    for i in range(num_req):
        bw_key = random.randint(0, 10)
        bw_data = random.randint(0, 10)
        req = (bw_key, bw_data)
        row = random.randint(0, nodes - 1)
        col = random.randint(0, nodes - 1)
        while row == col:
            col = random.randint(0, nodes - 1)
        traffic_matrix[row][col].append(req)
    return traffic_matrix


def generate_traffic_matrix_v2(nodes):
    # traffic_matrix = [[0 for _ in range(nodes)] for _ in range(nodes)]
    # for s in range(nodes):
    #     for d in range(nodes):
    #         if s == d: continue
    #         traffic_matrix[s][d] = random.randint(0, 10)
    traffic_matrix = [[0, 8, 5, 1, 6, 2], [1, 0, 1, 8, 4, 10], [10, 3, 0, 7, 0, 9], [2, 2, 7, 0, 8, 3], [0, 8, 8, 2, 0, 2], [5, 1, 8, 10, 7, 0]]
    return traffic_matrix


def cal_traffic_mat_req_num(traffic_matrix):
    res = [[len(j) for j in i] for i in traffic_matrix]
    return res


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')
    rwka_function()
