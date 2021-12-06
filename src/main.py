"""
The RWKA model for the PuLP modeller

Authors: LTH6696 2021
"""
import random

import numpy as np
import pandas as pd
from pulp import *


def rwka_function():
    prob = LpProblem("RWKA", LpMaximize)

    # Creates a list of all nodes
    nodes = [i for i in range(6)]

    # Creates a set of traffic matrix
    traffic_matrix = generate_traffic_matrix(len(nodes))

    # Initialise some inputs
    TR = [4 for _ in nodes]
    RR = [4 for _ in nodes]
    WL = 3
    wavelengths = [i for i in range(WL)]
    Pw = [[[1 for w in range(WL)] for n in nodes] for m in nodes]
    y = list(traffic_matrix.keys())
    C = 48
    # t = [t for yi in y for s in nodes for d in nodes for t in range(traffic_matrix[yi][s][d])]
    t = [i for i in range(11)]

    # A dictionary called 'v' is created to contain the referenced variables
    num_ligthpaths = LpVariable.dicts("LPs", (nodes, nodes), lowBound=0, upBound=None, cat=LpInteger)
    num_lps_on_wl = LpVariable.dicts("LPs_on_wl", (nodes, nodes, wavelengths), lowBound=0, upBound=1, cat=LpInteger)
    num_lps_routed_on_wl = LpVariable.dicts("phy_topo_route",
                                            (nodes, nodes, wavelengths, nodes, nodes),
                                            lowBound=0, upBound=1, cat=LpInteger)
    traffic_request_data = LpVariable.dicts('DataBwOneReq',
                                            (y, nodes, nodes, t, nodes, nodes),
                                            lowBound=0, upBound=1, cat=LpInteger)

    traffic_request_key = LpVariable.dicts('KeyBwOneReq',
                                           (y, nodes, nodes, t, nodes, nodes),
                                           lowBound=0, upBound=1, cat=LpInteger)
    successed_routed = LpVariable.dicts("SuccessRoute",
                                        (y, t, nodes, nodes),
                                        lowBound=0, upBound=1, cat=LpInteger)

    # The objective function is added to 'prob' first
    # prob += (lpSum(num_ligthpaths[i] for i in nodes))
    prob += (
        lpSum([yi * successed_routed[yi][ti][s][d] for yi in y for s in nodes for d in nodes for ti in range(traffic_matrix[yi][s][d])])
    )

    # The follow constraints are entered
    # On virtual-topology
    for i in nodes:
        prob += (
            lpSum([num_ligthpaths[i][j] for j in nodes]) <= TR[i],
            "constraint of transmitters {}".format(i)
        )

    for j in nodes:
        prob += (
            lpSum([num_ligthpaths[i][j] for i in nodes]) <= RR[j],
            "constraint of receivers {}".format(j)
        )

    for i in nodes:
        for j in nodes:
            prob += (
                lpSum([num_lps_on_wl[i][j][w] for w in wavelengths]) == num_ligthpaths[i][j],
                "constrain of wavelength {}{}".format(i, j)
            )

    # On physical route
    for i in nodes:
        for j in nodes:
            for w in wavelengths:
                for k in nodes:
                    if k != i and k != j:
                        prob += (
                            lpSum([num_lps_routed_on_wl[i][j][w][m][k] for m in nodes]) ==
                            lpSum([num_lps_routed_on_wl[i][j][w][k][n] for n in nodes]),
                            "lightpath continuity for {}{}{}{}".format(i, j, w, k)
                        )

    for i in nodes:
        for j in nodes:
            for w in wavelengths:
                prob += (
                    lpSum([num_lps_routed_on_wl[i][j][w][m][j] for m in nodes]) == num_lps_on_wl[i][j][w],
                    "incoming lightpath equal to total lightpaths on {}{}{}".format(i, j, w)
                )
                prob += (
                    lpSum([num_lps_routed_on_wl[i][j][w][i][n] for n in nodes]) == num_lps_on_wl[i][j][w],
                    "outgoing lightpath equal to total lightpaths on {}{}{}".format(i, j, w)
                )

    for i in nodes:
        for j in nodes:
            for w in wavelengths:
                prob += (
                    lpSum([num_lps_routed_on_wl[i][j][w][m][i] for m in nodes]) == 0,
                    "incoming lightpath is 0 on {}{}{}".format(i, j, w)
                )
                prob += (
                    lpSum([num_lps_routed_on_wl[i][j][w][j][n] for n in nodes]) == 0,
                    "outgoing lightpath is 0 on {}{}{}".format(i, j, w)
                )

    for w in wavelengths:
        for m in nodes:
            for n in nodes:
                prob += (
                    lpSum([num_lps_routed_on_wl[i][j][w][m][n] for i in nodes for j in nodes]) <= Pw[m][n][w],
                    "one fiber use one lightpath {}{}{}".format(w, m, n)
                )

    # On traffic bandwidth allocation
    for i in nodes:
        for j in nodes:
            prob += (
                lpSum(
                    [yi * lpSum([traffic_request_data[yi][s][d][t][i][j] for s in nodes for d in nodes for t in range(traffic_matrix[yi][s][d])]) for yi in y]
                ) <= num_ligthpaths[i][j] * C,
                "link bandwidth constraint {}{}".format(i, j)
            )

    for s in nodes:
        for d in nodes:
            for yi in y:
                for ti in range(traffic_matrix[yi][s][d]):
                    prob += (
                        lpSum([traffic_request_data[yi][s][d][ti][i][d] for i in nodes]) == successed_routed[yi][ti][s][d]
                    )

    for s in nodes:
        for d in nodes:
            for yi in y:
                for ti in range(traffic_matrix[yi][s][d]):
                    prob += (
                        lpSum([traffic_request_data[yi][s][d][ti][s][j] for j in nodes]) == successed_routed[yi][ti][s][d]
                    )

                    prob += (
                        lpSum([traffic_request_data[yi][s][d][ti][i][s] for i in nodes]) == 0
                    )

                    prob += (
                        lpSum([traffic_request_data[yi][s][d][ti][d][j] for j in nodes]) == 0
                    )

    for s in nodes:
        for d in nodes:
            for yi in y:
                for ti in range(traffic_matrix[yi][s][d]):
                    for k in nodes:
                        if k != s and k != d:
                            prob += (lpSum([traffic_request_data[yi][s][d][ti][i][k] for i in nodes]) ==
                                     lpSum([traffic_request_data[yi][s][d][ti][k][j] for j in nodes]))



    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # print("Status:", LpStatus[prob.status])
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)


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
    traffic_matrix = generate_traffic_matrix(6, 200)
    res = cal_traffic_mat_req_num(traffic_matrix)
    print(pd.DataFrame(res))