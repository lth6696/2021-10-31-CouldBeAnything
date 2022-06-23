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
from source.Result.ResultAnalysis import ResultAnalysisImpl, Results
from Algorithm.AlgorithmImp import *

BaseLine = 1  # Gbps
LightPathBandwidth = 100 * BaseLine  # 100Gbps


def simulate(nw: int, nconn: int, nl: int, solver='SLF-ML'):
    solvers = {'ILP-ML', 'ILP-SL', 'FF-ML', 'FF-SL', 'SLF-ML', 'MDPS-ML', 'SPF-ML'}
    if solver not in solvers:
        raise ValueError('Invalid solver \'{}\''.format(solver))
    operator, scheme = solver.split('-')
    scheme = True if scheme == 'ML' else False

    input_ = InputImp()
    input_.set_vertex_connection(path="./graphml/hexnet/hexnet.graphml", nw=nw, nl=nl, bandwidth=LightPathBandwidth)
    traffic_matrix = input_.get_traffic_matrix(nl=nl, nconn=nconn)

    start = time()
    if operator == 'ILP':
        adj_matrix = input_.get_adjacency_martix()
        level_matrix = input_.get_level_matrix()
        bandwidth_matrix = input_.get_bandwidth_matrix()
        result = IntegerLinearProgram().run(input_.MultiDiG, adj_matrix, level_matrix, bandwidth_matrix, traffic_matrix, multi_level=scheme)
    elif operator == 'MDPS':
        result = MetricDrivenPathSelect().simulate(input_.MultiDiG, traffic_matrix, multi_level=scheme, metric='MBF', weight=(0, 0, 1))
    elif operator == 'SPF':
        result = ShortestPathFirst().simulate(input_.MultiDiG, traffic_matrix, multi_level=scheme)
    else:
        operator = True if operator == 'SLF' else False
        result = SuitableLightpathFirst().simulate(input_.MultiDiG, traffic_matrix, slf=operator, multi_level=scheme)
    end = time()
    logging.info('{} - {} - The solver runs {:.3f} seconds.'.format(__file__, __name__, end-start))
    return result


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')

    nw = 2
    nl = 3
    Nconn = 16
    border = Nconn + 1
    repeat_times = 50
    # todo check1
    for nconn in range(Nconn, border):
        logging.info('{} - {} - nw: {} nconn: {} nl: {} start to run.'.format(__file__, __name__, nw, nconn, nl))
        # 每种情况运行repeat_times次，结果取平均
        results = Results()
        # todo a final result class need to be constructed
        for _ in range(repeat_times):
            # todo check2
            result = simulate(nw=nw, nconn=nconn, nl=nl, solver='ILP-ML')
            # todo check3
            result_analysis = ResultAnalysisImpl(result)
            throughput = result_analysis.analyze_throughput_for_each_level()
            hops = result_analysis.analyze_hop_for_each_level()
            lightpath_utilization = result_analysis.analyze_link_utilization_for_each_level(LightPathBandwidth)

            results.success_mapping_rate.append(result.traffic_mapping_success_rate)
            results.throughput.append(throughput[1])
            results.hops.append(hops[1])
            results.lightpath_utilization.append(lightpath_utilization[1])

        for attrs in ['success_mapping_rate', 'throughput', 'hops', 'lightpath_utilization']:
            all_run_result = (np.average(np.array(getattr(results, attrs)), axis=0).tolist())
            logging.info('{} - {} - nw: {} nconn: {} nl: {} done, the {} is {}.'.
                         format(__file__, __name__, nw, nconn, nl, attrs, all_run_result))

    # todo check4
    # data = {'success': {'wavelength': nw, 'level': nl, 'traffic': (1, border), 'SLF-ML': all_run_result}}
    # file = open('results_.json', 'w')
    # file.write(json.dumps(data))0
    # file.close()