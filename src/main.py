"""
The RWKA model for the PuLP modeller

Authors: LTH6696 2021
"""
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
    t = [t for yi in y for s in nodes for d in nodes for t in range(traffic_matrix[yi][s][d])]

    # A dictionary called 'v' is created to contain the referenced variables
    num_ligthpaths = LpVariable.dicts("LPs", (nodes, nodes), lowBound=0, upBound=None, cat=LpInteger)
    num_lps_on_wl = LpVariable.dicts("LPs_on_wl", (nodes, nodes, wavelengths), lowBound=0, upBound=1, cat=LpInteger)
    num_lps_routed_on_wl = LpVariable.dicts("phy_topo_route",
                                            (nodes, nodes, wavelengths, nodes, nodes),
                                            lowBound=0, upBound=1, cat=LpInteger)
    traffic_request_data = LpVariable.dicts('DataBwOneReq',
                                            (y, nodes, nodes, t, nodes, nodes),
                                            lowBound=0, upBound=1, cat=LpInteger)
    print(pd.DataFrame(traffic_matrix))
    print(t)
    print(pd.DataFrame(traffic_request_data[1][0][2]))
    traffic_request_key = LpVariable.dicts('KeyBwOneReq',
                                           (y, nodes, nodes, t, nodes, nodes),
                                           lowBound=0, upBound=1, cat=LpInteger)

    # The objective function is added to 'prob' first
    prob += (lpSum(num_ligthpaths[i] for i in nodes))

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
                    [yi * lpSum([traffic_request_data[s][d][yi][i][j] for s in nodes for d in nodes]) for yi in y]
                ) <= num_ligthpaths[i][j] * C,
                "link bandwidth constraint {}{}".format(i, j)
            )

    # for s in nodes:
    #     for d in nodes:
    #         for yi in y:
    #             for t in range(traffic_matrix[yi][s][d]):



    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # print("Status:", LpStatus[prob.status])
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)


def generate_traffic_matrix(nodes):
    security_level = [1, 3, 12, 48]
    traffic_matrix = {}
    for l in security_level:
        traffic_matrix[l] = [[np.random.randint(0, 10) for i in range(nodes)] for j in range(nodes)]
        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    traffic_matrix[l][i][j] = 0
    return traffic_matrix


if __name__ == '__main__':
    rwka_function()