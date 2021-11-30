"""
The RWKA model for the PuLP modeller

Authors: LTH6696 2021
"""
import numpy as np
from pulp import *


def rwka_function():
    prob = LpProblem("RWKA", LpMaximize)

    # Creates a list of all nodes
    vars = generate_variables(6)

    # vars = LpVariable(name="LigthPaths", lowBound=0, upBound=None, cat=LpInteger)
    v = LpVariable.dicts("Ligthpaths", vars, 0)

    # The objective function is added to 'prob' first
    prob += (lpSum(v[i] for i in vars))

    # The follow constraints are entered
    prob += (lpSum())

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    print("Status:", LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, "=", v.varValue)

def generate_variables(nodes):
    var = ['{}{}'.format(i, j) for i in range(nodes) for j in range(nodes)]
    return var

def generate_network_topology(nodes=None, conn_matrix=None):
    if nodes is None:
        nodes = 6
    if conn_matrix is None:
        conn_matrix = np.matrix('0 1 0 1 0 0; '
                                '1 0 1 1 0 0; '
                                '0 1 0 0 1 1; '
                                '1 1 0 0 1 0; '
                                '0 0 1 1 0 1; '
                                '0 0 1 0 1 0')
    return (nodes, conn_matrix)


if __name__ == '__main__':
    rwka_function()