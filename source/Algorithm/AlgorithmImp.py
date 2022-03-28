import pulp.pulp
from pulp import *

from .AlgorithmApi import Algorithm


class IntegerLinearProgram(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)


    def run(self, adj_matrix, traffic_matrix):
        prob = LpProblem("ServiceMapping", LpMaximize)
        solver = getSolver('CPLEX_CMD', timeLimit=10)

        row, col = adj_matrix.shape
        if row != col:
            logging.error('IntegerLinearProgram - run - Invalid adj_matrix.')
            return False
        nodes = [i for i in range(row)]

        # Variables
        Lamda = []  # list structure, L[src][dst][i][j][t]
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
                                temp_j.append(LpVariable("Lamda_{}_{}_{}_{}_{}".format(s, d, i, j, t),
                                                         lowBound=0,
                                                         upBound=1,
                                                         cat=LpInteger))
                        else:
                            temp_j.append(0)
                        temp_i.append(temp_j)
                    temp_dst.append(temp_i)
                temp_src.append(temp_dst)
            Lamda.append(temp_src)

        S = []  # list structure, S[src][dst].
        for s in range(row):
            temp_src = []
            for d in range(row):
                temp_src.append(LpVariable("Suc_{}_{}".format(s, d),
                                           lowBound=0, upBound=1, cat=LpInteger))
            S.append(temp_src)

        # Objective
        # The objective function is added to 'prob' first
        prob += (
            lpSum([S[s][d] for s in range(row) for d in range(row)])
        )

        # Constraints
        # continuity constraint
        for s in nodes:
            for d in nodes:
                prob += (
                        lpSum([
                            lpSum([Lamda[s][d][s][j][t] for t in range(adj_matrix[s][j])]) for j in nodes if
                            adj_matrix[s][j]
                        ]) == S[s][d]
                )
                prob += (
                        lpSum([
                            lpSum([Lamda[s][d][i][d][t] for t in range(adj_matrix[i][d])]) for i in nodes if
                            adj_matrix[i][d]
                        ]) == S[s][d]
                )
                for k in nodes:
                    if k != s and k != d:
                        prob += (
                                lpSum([
                                    lpSum([Lamda[s][d][i][k][t] for t in range(adj_matrix[i][k])]) for i in nodes if
                                    adj_matrix[i][k]
                                ]) ==
                                lpSum([
                                    lpSum([Lamda[s][d][k][j][t] for t in range(adj_matrix[k][j])]) for j in nodes if
                                    adj_matrix[k][j]
                                ])
                        )
                prob += (
                        lpSum([
                            lpSum([Lamda[s][d][i][s][t] for t in range(adj_matrix[i][s])]) for i in nodes if
                            adj_matrix[i][s]
                        ]) == 0
                )
                prob += (
                        lpSum([
                            lpSum([Lamda[s][d][d][j][t] for t in range(adj_matrix[d][j])]) for j in nodes if
                            adj_matrix[d][j]
                        ]) == 0
                )

        # maximum resource constraint
        for i in nodes:
            for j in nodes:
                if adj_matrix[i][j]:
                    for t in range(adj_matrix[i][j]):
                        prob += (
                                lpSum([Lamda[s][d][i][j][t] * traffic_matrix[s][d][0] for s in nodes for d in
                                       nodes]) <=
                                MaxLightpathRes[i][j] * 1 / (adj_weight[i][j][t] + 1)
                        )
                        prob += (
                                lpSum(
                                    [Lamda[s][d][i][j][t] * traffic_matrix[s][d][0] * traffic_matrix[s][d][1] for s
                                     in nodes for d in nodes]) <=
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

        logging.info("Status:{}".format(LpStatus[prob.status]))