import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import Input
from Algorithm.AlgorithmImp import IntegerLinearProgram

BaseLine = 1    # Mbps
LightPathBandwidth = 1000 * BaseLine


if __name__ == '__main__':
    input = Input.InputImp.InputImp()
    G = input.generate_topology(path='../graphml/nsfnet/nsfnet.graphml')
    adj_matrix = input.generate_adjacency_martix(G)
    lp_adj_matrix = input.generate_lightpath_adjacency_matrix(adj_matrix.copy(), nlp=4)
    lp_level_matrix = input.generate_lightpath_level_matrix(lp_adj_matrix, [1, 2, 3])
    bandwidth_matrix = input.generate_lightpath_bandwidth(lp_adj_matrix, LightPathBandwidth)
    print(pd.DataFrame(bandwidth_matrix))
    traffic_matrix = input.generate_traffic_matrix(nodes=[1, 2, 3])
    res = IntegerLinearProgram().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix)