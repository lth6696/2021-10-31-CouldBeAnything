from .AlgorithmApi import Algorithm

from pulp import *


class AlgorithmImp(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)

    def linear_integer_programming(self, sls, lls, s, t):
        # minimum bandwidth
        prob = LpProblem("ILP", LpMinimize)
        solver = getSolver('CPLEX_CMD', timeLimit=10)

        n_spine_switch = [i for i in range(len(sls))]
        n_leaf_switch = [j+len(sls) for j in range(len(lls))]
        n_switch = n_spine_switch + n_leaf_switch
        alpha = 10000

        # -------------------------- Variables --------------------------- #
        # xi that belongs to X indicates that whether the i switch belongs to the network.
        X = LpVariable.dicts('x', n_switch, lowBound=0, upBound=1, cat=LpInteger)

        # The k transmission speed from the m port of the i switch to the n port of the j switch.
        E = []
        for i in n_spine_switch:
            temp_i = []
            for m in range(sls[i].nport):
                temp_m = []
                for j in n_leaf_switch:
                    temp_j = []
                    for n in range(lls[j-len(sls)].nport):
                        temp_n = []
                        for k in range(len(s)):
                            v = LpVariable("E_{}_{}_{}_{}_{}".format(i, m, j, n, k),
                                           lowBound=0,
                                           upBound=1,
                                           cat=LpInteger)
                            temp_n.append(v)
                        temp_j.append(temp_n)
                    temp_m.append(temp_j)
                temp_i.append(temp_m)
            E.append(temp_i)

        # ---------------------------objective----------------------------- #
        prob += (
            lpSum([X[i]*sls[i].cost for i in n_spine_switch]) +
            lpSum([X[j]*lls[j-len(sls)].cost for j in n_leaf_switch])
        )

        # ---------------------------transfer speed------------------------ #
        for i in n_spine_switch:
            for m in range(sls[i].nport):
                prob += (
                    lpSum([s[k] * lpSum([lpSum([E[i][m][j-len(sls)][n][k] for n in range(lls[j-len(sls)].nport)]) for j in n_leaf_switch]) for k in range(len(s))])
                    <= sls[i].bandwidth
                )

                prob += (
                    lpSum([lpSum([lpSum([E[i][m][j-len(sls)][n][k] for k in range(len(s))]) for n in range(lls[j-len(sls)].nport)]) for j in n_leaf_switch])
                    <= 1
                )

        for j in n_leaf_switch:
            for n in range(lls[j-len(sls)].nport):
                prob += (
                    lpSum([s[k] * lpSum([lpSum([E[i][m][j-len(sls)][n][k] for m in range(sls[i].nport)]) for i in n_spine_switch]) for k in range(len(s))])
                    <= lls[j-len(sls)].bandwidth
                )

                prob += (
                    lpSum([lpSum([lpSum(E[i][m][j-len(sls)][n][k] for m in range(sls[i].nport)) for i in n_spine_switch]) for k in range(len(s))])
                    <= 1
                )

        # ----------------------------port--------------------------------- #
        for i in n_spine_switch:
            for j in n_leaf_switch:
                prob += (
                    lpSum([lpSum([lpSum(E[i][m][j-len(sls)][n][k] for k in range(len(s))) for n in range(lls[j-len(sls)].nport)]) for m in range(sls[i].nport)])
                    == 1
                )

        for i in n_spine_switch:
            prob += (
                lpSum([lpSum([lpSum([lpSum([E[i][m][j-len(sls)][n][k] for k in range(len(s))]) for m in range(sls[i].nport)]) for n in range(lls[j-len(sls)].nport)]) for j in n_leaf_switch])
                == lpSum([X[j] for j in n_leaf_switch])
            )

        for j in n_leaf_switch:
            prob += (
                lpSum([lpSum([lpSum([lpSum([E[i][m][j-len(sls)][n][k] for k in range(len(s))]) for n in range(lls[j-len(sls)].nport)]) for m in range(sls[i].nport)]) for i in n_spine_switch])
                == lpSum([X[i] for i in n_spine_switch])
            )

        # ----------------------------bandwidth---------------------------- #
        prob += (
            lpSum([lpSum([lpSum([lpSum([lpSum([E[i][m][j-len(sls)][n][k] * s[k] for k in range(len(s))]) for n in range(lls[j-len(sls)].nport)]) for m in range(sls[i].nport)]) for j in n_leaf_switch]) for i in n_spine_switch])
            >= t
        )

        # ----------------------------relationship------------------------- #
        for i in n_spine_switch:
            prob += (
                lpSum([lpSum([lpSum([lpSum([E[i][m][j-len(sls)][n][k] for k in range(len(s))]) for m in range(sls[i].nport)]) for n in range(lls[j-len(sls)].nport)]) for j in n_leaf_switch])
                >= X[i]
            )

            prob += (
                lpSum([lpSum([lpSum([lpSum([E[i][m][j-len(sls)][n][k] for k in range(len(s))]) for m in range(sls[i].nport)]) for n in range(lls[j-len(sls)].nport)]) for j in n_leaf_switch]) / alpha
                <= X[i]
            )

        for j in n_leaf_switch:
            prob += (
                lpSum([lpSum([lpSum([lpSum([E[i][m][j-len(sls)][n][k] for k in range(len(s))]) for n in range(lls[j-len(sls)].nport)]) for m in range(sls[i].nport)]) for i in n_spine_switch])
                >= X[j]
            )

            prob += (
                lpSum([lpSum([lpSum([lpSum([E[i][m][j - len(sls)][n][k] for k in range(len(s))]) for n in range(lls[j - len(sls)].nport)]) for m in range(sls[i].nport)]) for i in n_spine_switch]) / alpha
                <= X[j]
            )

        prob.solve()

        print("Status:", LpStatus[prob.status])
        # for v in prob.variables():
        #     print(v.name, "=", v.varValue)