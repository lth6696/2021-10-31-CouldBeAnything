import warnings
import logging
import logging.config

import numpy as np
import networkx as nx
import pandas as pd
from time import time

from source.input.InputImp import InputImp
from source.algorithm.AlgorithmImp import *

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

BaseLine = 1  # 1 Gb/s
LightPathBandwidth = 100 * BaseLine  # 100 Gb/s


def simulate(nw: int, ntm: int, nl: int, method: str, topo_file: str):
    """
    本方法按顺序执行算法步骤
    :param nw: int, 波长数
    :param ntm: int, 流量矩阵数
    :param nl: int, 安全等级数
    :param method: str, 求解方法
    :param topo_file: str, 拓扑文件路径
    :return: obj, 仿真结果
    """
    allowed_methods = {'ILP-LBMS', 'ILP-LSMS', 'SASMA-LBMS', 'SASMA-LSMS'}
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
    else:
        sasma = SecurityAwareServiceMappingAlgorithm()
        result = sasma.solve(input_.MultiDiG, traffic_matrix, scheme=scheme)
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
    Method = 'ILP-LSMS'     # 共有四种求解方式 {'ILP-LBMS', 'ILP-LSMS', 'SASMA-LBMS', 'SASMA-LSMS'}
    TopoFile = "./graphml/hexnet/hexnet.graphml"
    SaveFile = 'result_matrix.npy'

    # 仿真
    metrics = {'mapping_rate', 'throughput', 'ave_hops', 'ave_link_utilization', 'ave_level_deviation'}
    result_matrix = np.zeros(shape=(Nmatrix, RepeatTimes, len(metrics)))
    # for K in range(1, Nmatrix+1):
    for K in [12, 16]:
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
                              topo_file=TopoFile)
            result_matrix[K-1][i] = [result.mapping_rate,
                                     result.throughput,
                                     result.ave_hops,
                                     result.ave_link_utilization,
                                     result.ave_level_deviation]
        print('----------------{}-------------------'.format(K))
        print(pd.DataFrame(np.mean(result_matrix[K-1], axis=0)))
    # 保存结果矩阵
    np.save(SaveFile, result_matrix)
