import pandas as pd
import logging
import logging.config
from time import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import Input
from Algorithm.AlgorithmImp import *

BaseLine = 1    # Gbps
LightPathBandwidth = 100 * BaseLine     # 100Gbps


if __name__ == '__main__':
    NWaveL = 4
    NConn = 2
    levels = [i+1 for i in range(3)]

    logging.config.fileConfig('logconfig.ini')
    input = Input.InputImp.InputImp()
    G = input.generate_topology(path='../graphml/nsfnet/nsfnet.graphml')
    adj_matrix = input.generate_adjacency_martix(G)
    nodes, _ = adj_matrix.shape
    lp_adj_matrix = input.generate_lightpath_adjacency_matrix(adj_matrix.copy(), nlp=NWaveL)
    lp_level_matrix = input.generate_lightpath_level_matrix(lp_adj_matrix, levels)
    bandwidth_matrix = input.generate_lightpath_bandwidth(lp_adj_matrix, LightPathBandwidth)
    traffic_matrix = input.generate_traffic_matrix(nodes=[i for i in range(nodes)], levels=levels, nconn=NConn)

    start = time()
    # res = IntegerLinearProgram().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix)
    # res = Heuristic().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix, multi_level=True)
    res = SuitableLightpathFirst().simulate(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix, slf=False, multi_level=False)
    end = time()
    print("Running time is {}".format(end - start))
