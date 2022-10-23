import logging.config
import warnings

import pandas as pd
import numpy as np

import result.output as op
from input import network, traffic
from solver import heuristic

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')
    # 仿真参数
    repeat_times = 50
    ips_per_gigabyte = 1000
    # 初始化网络拓扑
    topology_obj = network.Topology()
    # 仿真
    data = [['latency(s)', 'hop', 'routed services', 'success rate', 'throughput',
             'com_utl', 'sto_utl', 'bandwidth_utl', 'cost']]
    for K in [1, 2, 3, 4, 5]:
        result_matrix = np.empty(shape=(len(data[0]), repeat_times))
        for i in range(repeat_times):
            logging.info("It's running the {}th matrices in the {}th times.".format(K, i))
            graph = topology_obj.generate_topology()
            # 初始化业务矩阵
            traffic_matrix_obj = traffic.TrafficMatrix(graph)
            traffic_matrices = traffic_matrix_obj.generate_traffic_matrices(K)
            # 求解
            cara = heuristic.ComputingAwareRoutingAllocating()
            path = cara.run(graph, traffic_matrices)
            solution = op.Solution()
            solution.init(path=path, graph=graph, traffic_matrix=traffic_matrices)
            res = op.Performance()
            result_matrix[0][i] = res.get_latency(solution, ips_per_gigabyte)  # latency(second)
            result_matrix[1][i] = res.get_hop(solution)  # hop
            result_matrix[2][i] = res.get_routed_service(solution)  # routed services
            result_matrix[3][i] = res.get_success_rate(solution)  # success rate
            result_matrix[4][i] = res.get_throughput(solution)  # throughput
            result_matrix[5][i] = res.get_compute_utilization(solution)  # com_utl
            result_matrix[6][i] = res.get_storage_utilization(solution)  # sto_utl
            result_matrix[7][i] = res.get_link_utilization(solution)  # bandwidth_utl
            result_matrix[8][i] = res.get_cost(solution)  # cost
        data.append(('K={}'.format(K), np.average(result_matrix, axis=1)))
        print("{}K = {}{}".format('-'*50, K, '-'*50))
        print(pd.DataFrame(np.reshape(data[-1][1], newshape=(1, len(data[0]))), columns=data[0]))
    np.save('result.npy', data)
