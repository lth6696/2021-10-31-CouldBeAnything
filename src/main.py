"""
The RWKA model for the PuLP modeller

Authors: LTH6696 2021
"""
from pulp import *


def rwka_function():
    prob = LpProblem("RWKA", LpMaximize)

    # Creates a list of all nodes
    nodes = [i for i in range(6)]

    # Initialise some constants
    TR = [4 for _ in nodes]
    RR = [4 for _ in nodes]
    WL = 3
    wavelengths = [i for i in range(WL)]

    # A dictionary called 'v' is created to contain the referenced variables
    num_ligthpaths = LpVariable.dicts("LPs", (nodes, nodes), lowBound=0, upBound=None, cat=LpInteger)
    num_lps_on_wl = LpVariable.dicts("LPs_on_wl", (nodes, nodes, wavelengths), lowBound=0, upBound=1, cat=LpInteger)

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
    

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    print("Status:", LpStatus[prob.status])
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)


if __name__ == '__main__':
    rwka_function()