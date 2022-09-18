import networkx as nx
import pandas as pd
import logging
import logging.config
from time import time
import numpy as np
import json

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from source.input.InputImp import InputImp
from source.result.ResultPlot import ResultPresentation
from source.result.ResultAnalysis import ResultAnalysisImpl, Results
from source.algorithm.AlgorithmImp import *

BaseLine = 1  # Gbps
LightPathBandwidth = 100 * BaseLine  # 100Gbps


def simulate(nw: int, ntm: int, nl: int, method: str):
    """
    本方法按顺序执行算法步骤
    :param nw: int, 波长数
    :param ntm: int, 流量矩阵数
    :param nl: int, 安全等级数
    :param method: str, 求解方法
    :return: obj, 仿真结果
    """
    allowed_methods = {'ILP-LBMS', 'ILP-LSMS', 'SASMA-LBMS', 'SASMA-LSMS'}
    if method not in allowed_methods:
        raise ValueError('Invalid solver \'{}\''.format(method))
    solver, scheme = method.split('-')

    # 初始化模型输入
    input_ = InputImp()
    input_.set_vertex_connection(path="./graphml/nsfnet/nsfnet.graphml", nw=nw, nl=nl, bandwidth=LightPathBandwidth)
    traffic_matrix = input_.get_traffic_matrix(nl=nl, nconn=ntm)

    start = time()
    if solver == 'ILP':
        adj_matrix = input_.get_adjacency_martix()
        level_matrix = input_.get_level_matrix()
        bandwidth_matrix = input_.get_bandwidth_matrix()
        result = IntegerLinearProgram().run(input_.MultiDiG, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, multi_level=scheme)
    else:
        solver = True if solver == 'SASMA' else False
        sasma = SecurityAwareServiceMappingAlgorithm()
        result = sasma.solve(input_.MultiDiG, traffic_matrix, slf=solver, multi_level=scheme)
    end = time()
    logging.info('{} - {} - The solver runs {:.3f} seconds.'.format(__file__, __name__, end-start))
    return result


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')

    # 初始化参数
    Nwavelength = 4     # 波长数
    Nlevel = 3          # 安全等级数
    Nmatrix = 40        # 流量矩阵数
    RepeatTimes = 50    # 重复实验次数
    Method = 'SASMA-LSMS'   # 默认为启发式算法-不可越级模式

    # 仿真
    save_data = defaultdict(list)
    for K in range(1, Nmatrix+1):
        logging.info('{} - {} - Simulation sets {} wavelengths, {}/{} levels and {}/{} matrices.'
                     .format(__file__, __name__,
                             Nwavelength,
                             L if 'L' in dir() else Nlevel, Nlevel,
                             K if 'K' in dir() else Nmatrix, Nmatrix))
        # 实验重复RepeatTimes次，结果取平均
        results = Results()
        for _ in range(RepeatTimes):
            result = simulate(nw=Nwavelength,
                              ntm=K if 'K' in dir() else Nmatrix,
                              nl=L if 'L' in dir() else Nlevel,
                              method=Method)
            result_analysis = ResultAnalysisImpl(result)
            # throughput = result_analysis.analyze_throughput_for_each_level()
            # hops = result_analysis.analyze_hop_for_each_level()
            # lightpath_utilization = result_analysis.analyze_link_utilization_for_each_level(LightPathBandwidth)
            # level_deviation = result_analysis.analyze_deviation_for_each_level()

            # results.success_mapping_rate.append(result.traffic_mapping_success_rate)
            # results.success_mapping_rate_each_level.append(result.traffic_mapping_success_rate_each_level)
            # results.throughput.append(np.sum(throughput[1]))
            # results.hops.append(np.mean(hops[1]))
            # results.lightpath_utilization.append(np.mean(lightpath_utilization[1]))
            # results.level_deviation.append(level_deviation[1])
        #
        # for attrs in ['success_mapping_rate', 'success_mapping_rate_each_level', 'throughput', 'hops', 'lightpath_utilization']:
        #     all_run_result = (np.average(np.array(getattr(results, attrs)), axis=0).tolist())
        #     logging.info('{} - {} - nw: {} nconn: {} nl: {} done, the {} is {}.'.
        #                  format(__file__, __name__, nw, nconn, Nlevel, attrs, all_run_result))
        #     save_data[attrs].append(all_run_result)

    # # todo check4
    # data = {'success': {'wavelength': nw, 'level': Nlevel, 'traffic': (Nconn, border), solver: save_data}}
    # file = open('temp_.json', 'w')
    # file.write(json.dumps(data))
    # file.close()