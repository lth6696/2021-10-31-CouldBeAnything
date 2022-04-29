import pandas as pd
import logging
import logging.config
from time import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from source.Input.InputImp import InputImp
from Algorithm.AlgorithmImp import *

BaseLine = 1  # Gbps
LightPathBandwidth = 100 * BaseLine  # 100Gbps

if __name__ == '__main__':
    NWaveL = 4
    NConn = 4
    Nlevel = 5

    logging.config.fileConfig('logconfig.ini')
    input = InputImp()
    input.set_vertex_connection(path="./graphml/nsfnet/nsfnet.graphml", nw=NWaveL, nl=Nlevel, bandwidth=LightPathBandwidth)
    lp_adj_matrix = input.get_adjacency_martix()
    lp_level_matrix = input.get_level_matrix()
    bandwidth_matrix = input.get_bandwidth_matrix()
    traffic_matrix = input.get_traffic_matrix(nl=Nlevel, nconn=NConn)

    start = time()
    # res = IntegerLinearProgram().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix, multi_level=True)
    # res = Heuristic().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix, multi_level=True)
    res = SuitableLightpathFirst().simulate(input.MultiDiG, traffic_matrix, slf=True, multi_level=True)
    end = time()
    print("Running time is {}".format(end - start))
