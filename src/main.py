"""
The RWKA model for the PuLP modeller

Authors: LTH6696 2021
"""
import random

import numpy as np
import pandas as pd
from pulp import *


def rwka_function():
    global S
    prob = LpProblem("BD", LpMaximize)

    # Creates a list of all nodes
    nodes = [i for i in range(6)]

    # Creates adjust matrix
    adj_matrix = generate_adj_matrix(len(nodes))

    # Creates traffic matrix
    traffic_matrix = generate_traffic_matrix(len(nodes), 10)
    traffic_mat_num = cal_traffic_mat_req_num(traffic_matrix)

    # Inputs
    ReservedKeyBw = 10 * adj_matrix
    ReservedDataBw = 10 * adj_matrix

    # Variables
    # K indicates whether the key bandwidth is successfully allocated.
    K = []      # list structure, K[src][dst][req][i][j]
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_dst = []
            for n in range(traffic_mat_num[s][d]):
                temp_req = []
                for i in nodes:
                    temp_i = []
                    for j in nodes:
                        if adj_matrix[i][j]:
                            temp_i.append(LpVariable("KeyBwSucAlloc_{}_{}_{}_{}_{}".format(s, d, n, i, j),
                                                     lowBound=0,
                                                     upBound=1,
                                                     cat=LpInteger))
                    temp_req.append(temp_i)
                temp_dst.append(temp_req)
            temp_src.append(temp_dst)
        K.append(temp_src)

    # S indicates whether the request is successfully allocated.
    S = []      # list structure, S[src][dst][req], the number of var(s) is equal to the number of req(s).
    for s in nodes:
        temp_src = []
        for d in nodes:
            temp_dst = []
            for n in range(traffic_mat_num[s][d]):
                temp_dst.append(LpVariable("SuccessAllocate_{}_{}_{}".format(s, d, n),
                                           lowBound=0, upBound=1))
            temp_src.append(temp_dst)
        S.append(temp_src)

    # The objective function is added to 'prob' first
    prob += (
        lpSum([S[s][d][n] for s in nodes for d in nodes for n in range(traffic_mat_num[s][d])])
    )

    # The follow constraints are entered
    for s in nodes:
        for d in nodes:
            if traffic_mat_num[s][d]:
                for n in range(traffic_mat_num[s][d]):
                    val = lpSum(j for i in K[s][d][n] for j in i)
                    prob += (
                        val == S[s][d][n]
                    )


    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    print("Status:", LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, "=", v.varValue)


def generate_adj_matrix(nodes):
    # row represents src node, col represents dst node.
    adj_matrix = [[random.randint(0, 1) if i != j else 0 for j in range(nodes)] for i in range(nodes)]
    return adj_matrix


def generate_traffic_matrix(nodes, num_req):
    traffic_matrix = [[[] for _ in range(nodes)] for _ in range(nodes)]
    for i in range(num_req):
        bw_key = random.randint(0, 10)
        bw_data = random.randint(0, 10)
        req = (bw_key, bw_data)
        row = random.randint(0, nodes-1)
        col = random.randint(0, nodes-1)
        while row == col:
            col = random.randint(0, nodes - 1)
        traffic_matrix[row][col].append(req)
    return traffic_matrix


def cal_traffic_mat_req_num(traffic_matrix):
    res = [[len(j) for j in i] for i in traffic_matrix]
    return res


if __name__ == '__main__':
    rwka_function()
