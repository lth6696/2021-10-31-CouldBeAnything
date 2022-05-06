import networkx as nx
import pandas as pd
import logging
import logging.config
from time import time
import numpy as np
import json

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

from source.Input.InputImp import InputImp
from source.Result.ResultPlot import ResultPresentation
from source.Result.ResultAnalysis import ResultAnalysisImpl
from Algorithm.AlgorithmImp import *

BaseLine = 1  # Gbps
LightPathBandwidth = 100 * BaseLine  # 100Gbps


def simulate(nw: int, nconn: int, nl: int, solver='SLF-ML'):
    solvers = {'ILP-ML', 'ILP-SL', 'FF-ML', 'FF-SL', 'SLF-ML', 'SLF-SL'}
    if solver not in solvers:
        raise ValueError('Invalid solver \'{}\''.format(solver))
    operator, scheme = solver.split('-')
    scheme = True if scheme == 'ML' else False

    input_ = InputImp()
    input_.set_vertex_connection(path="./graphml/nsfnet/nsfnet.graphml", nw=nw, nl=nl, bandwidth=LightPathBandwidth)
    traffic_matrix = input_.get_traffic_matrix(nl=nl, nconn=nconn)

    start = time()
    if operator == 'ILP':
        adj_matrix = input_.get_adjacency_martix()
        level_matrix = input_.get_level_matrix()
        bandwidth_matrix = input_.get_bandwidth_matrix()
        result = IntegerLinearProgram().run(adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, multi_level=scheme)
    else:
        operator = True if operator == 'SLF' else False
        result = SuitableLightpathFirst().simulate(input_.MultiDiG, traffic_matrix, slf=operator, multi_level=scheme)
    end = time()
    logging.info('{} - {} - The solver runs {:.3f} seconds.'.format(__file__, __name__, end-start))
    return result


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')

    nw = 4
    nl = 3
    nconn = 4
    border = 41
    all_run_result = []
    for nconn in range(1, border):
        logging.info('{} - {} - nw: {} nconn: {} nl: {} start to run.'.format(__file__, __name__, nw, nconn, nl))
        # 每种情况运行10次，结果取平均
        one_run_result = [[i+1 for i in range(nl)]]
        for _ in range(10):
            result = simulate(nw=nw, nconn=nconn, nl=nl, solver='FF-ML')
            res_ana = ResultAnalysisImpl(result).analyze_link_utilization_for_each_level(LightPathBandwidth)
            if one_run_result[0] != res_ana[0]:
                logging.error('{} - {} - nw: {} nconn: {} nl: {} misses results: {} - {}.'.format(__file__, __name__,
                                                                                                  nw, nconn, nl,
                                                                                                  one_run_result[0],
                                                                                                  res_ana[0]))
                raise RuntimeError('Missing results.')
            one_run_result.append(res_ana[1])
        all_run_result.append(np.average(np.array(one_run_result)[1:, :], axis=0).tolist())
        logging.info('{} - {} - nw: {} nconn: {} nl: {} done, the result is {}.'.format(__file__, __name__, nw, nconn, nl, all_run_result[-1]))

    print(all_run_result)
    data = {'utilization': {'wavelength': nw, 'level': nl, 'traffic': (1, border), 'SLF': all_run_result}}
    file = open('results_.json', 'w')
    file.write(json.dumps(data))
    file.close()