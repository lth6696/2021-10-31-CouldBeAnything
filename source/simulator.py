import pandas as pd
import logging
import logging.config
from time import time
import numpy as np

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from source.Input.InputImp import InputImp
from source.Result.ResultPlot import ResultPresentation
from source.Result.ResultAnalysis import ResultAnalysisImpl
from Algorithm.AlgorithmImp import *

BaseLine = 1  # Gbps
LightPathBandwidth = 100 * BaseLine  # 100Gbps


def simulate_once():
    NWaveL = 4
    NConn = 4
    Nlevel = 11

    input = InputImp()
    input.set_vertex_connection(path="./graphml/nsfnet/nsfnet.graphml", nw=NWaveL, nl=Nlevel,
                                bandwidth=LightPathBandwidth)
    traffic_matrix = input.get_traffic_matrix(nl=Nlevel, nconn=NConn)

    start = time()
    result = SuitableLightpathFirst().simulate(input.MultiDiG, traffic_matrix, slf=True, multi_level=True)
    end = time()
    print(result.traffic_mapping_success_rate)
    print("Running time is {}".format(end - start))
    result_matrix = ResultAnalysisImpl().analysis_traffic_block_rate_under_different_situation(result)
    print([np.mean(result_matrix[:][col]) for col in range(len(result_matrix[0]))])
    return []


def simulate_gradient():
    NWaveL = 4
    NConn = 4
    Nlevel = 3

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

    level_change_res = simulate_once()
    # ResultPresentation().plot_line(level_change_res)

