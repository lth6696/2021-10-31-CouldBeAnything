import pandas as pd
import logging
import logging.config
from time import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from source.Input.InputImp import InputImp
from source.Result.plot_res import ResultPresentation
from Algorithm.AlgorithmImp import *

BaseLine = 1  # Gbps
LightPathBandwidth = 100 * BaseLine  # 100Gbps


def simulate_once():
    NWaveL = 4
    NConn = 4
    Nlevel = 5

    input = InputImp()
    input.set_vertex_connection(path="./graphml/nsfnet/nsfnet.graphml", nw=NWaveL, nl=Nlevel,
                                bandwidth=LightPathBandwidth)
    traffic_matrix = input.get_traffic_matrix(nl=Nlevel, nconn=NConn)

    start = time()
    SuitableLightpathFirst().simulate(input.MultiDiG, traffic_matrix, slf=True, multi_level=True)
    end = time()
    print("Running time is {}".format(end - start))

def simulate_gradient():
    NWaveL = 4
    NConn = 4

    level_change_res = []
    for Nlevel in range(1, 20):
        input = InputImp()
        input.set_vertex_connection(path="./graphml/nsfnet/nsfnet.graphml", nw=NWaveL, nl=Nlevel,
                                    bandwidth=LightPathBandwidth)
        traffic_matrix = input.get_traffic_matrix(nl=Nlevel, nconn=NConn)

        # lp_adj_matrix = input.get_adjacency_martix()
        # lp_level_matrix = input.get_level_matrix()
        # bandwidth_matrix = input.get_bandwidth_matrix()

        start = time()
        # res = IntegerLinearProgram().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix, multi_level=True)
        # res = Heuristic().run(lp_adj_matrix, lp_level_matrix, bandwidth_matrix, traffic_matrix, multi_level=True)
        res = SuitableLightpathFirst().simulate(input.MultiDiG, traffic_matrix, slf=True, multi_level=True)
        end = time()
        print("Running time is {}".format(end - start))

        level_change_res.append(res)
    return level_change_res


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')

    level_change_res = simulate_gradient()
    ResultPresentation().plot_line(level_change_res)

