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
    solvers = {'ILP-ML', 'ILP-SL', 'FF-ML', 'FF-SL', 'SLF-ML'}
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

    nw = 4
    Nlevel = 3
    Nconn = 1
    border = Nconn + 40
    repeat_times = 50
    solver = 'SLF-ML'
    # todo check1
    save_data = defaultdict(list)
    for nconn in range(Nconn, border):
        logging.info('{} - {} - nw: {} nconn: {} nl: {} start to run.'.format(__file__, __name__, nw, nconn, Nlevel))
        # 每种情况运行repeat_times次，结果取平均
        results = Results()
        for _ in range(repeat_times):
            # todo check2
            result = simulate(nw=nw, nconn=nconn, nl=Nlevel, solver=solver)
            # todo check3
            result_analysis = ResultAnalysisImpl(result)
            throughput = result_analysis.analyze_throughput_for_each_level()
            hops = result_analysis.analyze_hop_for_each_level()
            lightpath_utilization = result_analysis.analyze_link_utilization_for_each_level(LightPathBandwidth)
            level_deviation = result_analysis.analyze_deviation_for_each_level()

            results.success_mapping_rate.append(result.traffic_mapping_success_rate)
            results.success_mapping_rate_each_level.append(result.traffic_mapping_success_rate_each_level)
            results.throughput.append(np.sum(throughput[1]))
            results.hops.append(np.mean(hops[1]))
            results.lightpath_utilization.append(np.mean(lightpath_utilization[1]))
            results.level_deviation.append(level_deviation[1])

        for attrs in ['success_mapping_rate', 'success_mapping_rate_each_level', 'throughput', 'hops', 'lightpath_utilization']:
            all_run_result = (np.average(np.array(getattr(results, attrs)), axis=0).tolist())
            logging.info('{} - {} - nw: {} nconn: {} nl: {} done, the {} is {}.'.
                         format(__file__, __name__, nw, nconn, Nlevel, attrs, all_run_result))
            save_data[attrs].append(all_run_result)

    # todo check4
    data = {'success': {'wavelength': nw, 'level': Nlevel, 'traffic': (Nconn, border), solver: save_data}}
    file = open('temp_.json', 'w')
    file.write(json.dumps(data))
    file.close()