import warnings
import logging
import logging.config

import numpy as np
import pandas as pd
from time import time

from input.InputImp import InputImp
from algorithm.AlgorithmImp import *

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

BaseLine = 1  # 1 Gb/s
LightPathBandwidth = 100 * BaseLine  # 100 Gb/s


def simulate(nw: int, ntm: int, nl: int, method: str, topo_file: str, weights: tuple):
    """
    本方法按顺序执行算法步骤
    :param nw: int, 波长数
    :param ntm: int, 流量矩阵数
    :param nl: int, 安全等级数
    :param method: str, 求解方法
    :param topo_file: str, 拓扑文件路径
    :param weights: tuple, 添加边缘时，多指标权重，权重和为 1
    :return: obj, 仿真结果
    """
    allowed_methods = {'ILP-LBMS', 'ILP-LSMS', 'LFEL-LBMS', 'LFEL-LSMS', 'EO-LBMS', 'EO-LSMS'}
    if method not in allowed_methods:
        raise ValueError('Invalid solver \'{}\''.format(method))
    solver, scheme = method.split('-')

    # 初始化模型输入
    input_ = InputImp()
    input_.set_vertex_connection(path=topo_file, nw=nw, nl=nl, bandwidth=LightPathBandwidth)
    traffic_matrix = input_.get_traffic_matrix(nl=nl, ntm=ntm)

    start = time()
    if solver == 'ILP':
        adj_matrix = input_.get_adjacency_martix()
        level_matrix = input_.get_level_matrix()
        bandwidth_matrix = input_.get_bandwidth_matrix()
        result = IntegerLinearProgram().run(input_.MultiDiG, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, scheme)
    elif solver == 'LFEL':
        result = LayerFirstEdgeLast().solve(input_.MultiDiG, traffic_matrix, weights, scheme)
    elif solver == 'EO':
        result = EdgeOnly().solve(input_.MultiDiG, traffic_matrix, weights, scheme)
    else:
        result = None
        logging.error('{} - {} - Inputting wrong scheme {}.'.format(__file__, __name__, scheme))
    end = time()
    logging.info('{} - {} - The solver runs {:.3f} seconds.'.format(__file__, __name__, end-start))
    return result


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')

    # 初始化参数
    Nwavelength = 4     # 波长数
    Nlevel = 3          # 安全等级数
    Nmatrix = 25        # 流量矩阵数
    RepeatTimes = 50    # 重复实验次数
    Method = 'EO-LSMS'     # 共有多种求解方式 {'ILP-LBMS', 'ILP-LSMS', 'LFEL-LBMS', 'LFEL-LSMS', 'EO-LBMS', 'EO-LSMS'}
    MetricWeights = (0, 0, 1)    # 指标有四种：1、跨越等级 2、占用带宽 3、抢占带宽比例
    TopoFile = "./graphml/nsfnet/nsfnet.graphml"
    SaveFile = 'result_matrix.npy'

    # 仿真
    metrics = ('mapping_rate', 'service_throughput', 'network_throughput', 'req_bandwidth', 'ave_hops', 'ave_link_utilization', 'ave_level_deviation')
    result_matrix = np.zeros(shape=(Nmatrix, RepeatTimes, len(metrics)))
    for K in range(1, Nmatrix+1):
        logging.info('{} - {} - Simulation sets {} wavelengths, {}/{} levels and {}/{} matrices.'
                     .format(__file__, __name__,
                             Nwavelength,
                             L if 'L' in dir() else Nlevel, Nlevel,
                             K if 'K' in dir() else Nmatrix, Nmatrix))
        # 实验重复RepeatTimes次，结果取平均
        for i in range(RepeatTimes):
            result = simulate(nw=Nwavelength,
                              ntm=K if 'K' in dir() else Nmatrix,
                              nl=L if 'L' in dir() else Nlevel,
                              method=Method,
                              topo_file=TopoFile,
                              weights=MetricWeights)
            result_matrix[K-1][i] = [result.mapping_rate,
                                     result.service_throughput,
                                     result.network_throughput,
                                     result.req_bandwidth,
                                     result.ave_hops,
                                     result.ave_link_utilization,
                                     result.ave_level_deviation]
        print('{}K={}{}'.format('-'*60, K, '-'*60))
        print(pd.DataFrame([metrics, np.mean(result_matrix[K-1], axis=0)]))
    # 保存结果矩阵
    np.save(SaveFile, result_matrix)
