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
    Nlevel = 3

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
    result_matrix = np.array(result_matrix)
    print(pd.DataFrame(result_matrix))
    print([np.mean(result_matrix[:, col]) for col in range(result_matrix.shape[1])])
    result_matrix = ResultAnalysisImpl().analysis_lightpath_level_distribution(input.MultiDiG)
    print(pd.DataFrame(result_matrix))
    print(pd.DataFrame(result_matrix).sum())
    return []


def simulate_gradient(slf=True):
    NWaveL = 4
    NConn = 4
    Nlevel = 3

    level_change_res = []
    block_distribution = {'sum': [], 'mean': []}
    different_level_distribute_matrix = [[0 for _ in range(40)] for _ in range(38)]
    for NConn in range(2, 40, 1):
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
        res = SuitableLightpathFirst().simulate(input.MultiDiG, traffic_matrix, slf, multi_level=True)
        end = time()
        print("{} {} - Running time is {}".format(NConn, Nlevel, end - start))
        level_change_res.append(res.traffic_mapping_success_rate)
        # 获取阻塞率数据
        result_matrix = ResultAnalysisImpl().analysis_traffic_block_rate_under_different_situation(res)
        result_matrix = np.array(result_matrix)
        # 绘制 每次迭代多等级某一阻塞情况下的柱状图
        for row in result_matrix:
            different_level_distribute_matrix[NConn-2][int(row[0])] = row[2]   # row[1] - 0x01 row[2] - 0x02
        # 绘制 阻塞率vs阻塞情况(对应各个等级) 柱状图
        # ResultPresentation().plot_block_distribution_under_different_situation(result_matrix, situations=situations)
        # 绘制 阻塞率vs阻塞情况（每次迭代平均走势） 柱状图
        block_distribution['mean'].append([np.mean(result_matrix[:, col]) for col in range(result_matrix.shape[1])])
        # 绘制 阻塞率vs阻塞情况（每次迭代总体走势） 柱状图
        block_distribution['sum'].append([np.sum(result_matrix[:, col]) for col in range(result_matrix.shape[1])])
    ResultPresentation().plot_block_distribution_under_different_situation(block_distribution['sum'],
                                                                           situations=['0x01', '0x02'])
    ResultPresentation().plot_block_distribution_under_different_situation(np.array(different_level_distribute_matrix)[:, :4])

    return level_change_res


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')

    level_change_res = []
    level_change_res.append(simulate_gradient(slf=False))
    print('--------------------------------------------------')
    # level_change_res.append(simulate_gradient(slf=True))
    ResultPresentation().plot_multi_lines(level_change_res)
    # simulate_once()

