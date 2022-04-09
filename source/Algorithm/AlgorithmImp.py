import pulp.pulp
from pulp import *

from .AlgorithmApi import Algorithm


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

        # The problem is solved using PuLP's choice of Solver
        prob.solve(solver=solver)
        # for v in prob.variables():
        #     if v.varValue == 1:
        #         print("{} = {}".format(v.name, v.varValue))

        logging.info("Status:{}".format(LpStatus[prob.status]))
        return prob