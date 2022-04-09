import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import Input
from Algorithm.AlgorithmImp import IntegerLinearProgram

BaseLine = 1    # Gbps
LightPathBandwidth = 100 * BaseLine     # 100Gbps


if __name__ == '__main__':
    input = Input.InputImp.InputImp()
    G = input.generate_topology(path='../graphml/nsfnet/nsfnet.graphml')
    adj_matrix = input.generate_adjacency_martix(G)
    nodes, _ = adj_matrix.shape
    lp_adj_matrix = input.generate_lightpath_adjacency_matrix(adj_matrix.copy(), nlp=4)
    lp_level_matrix = input.generate_lightpath_level_matrix(lp_adj_matrix, [1, 2, 3])
    bandwidth_matrix = input.generate_lightpath_bandwidth(lp_adj_matrix, LightPathBandwidth)
    print(pd.DataFrame(bandwidth_matrix))
    print(pd.DataFrame(lp_adj_matrix))
    print('level -----------------------\n', pd.DataFrame(lp_level_matrix))
    traffic_matrix = input.generate_traffic_matrix(nodes=[i for i in range(nodes)], nconn=4)
    res = IntegerLinearProgram().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix)