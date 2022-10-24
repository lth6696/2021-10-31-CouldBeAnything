import logging.config
import warnings

import pandas as pd
import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt

import result.output as op
from input import network, traffic
from solver import problem_defination

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')
    # 初始化网络拓扑
    repeat_times = 10
    ips_per_gigabyte = 1000
    NIND = 50
    MAXGEN = 200
    topology_obj = network.Topology()
    data = [['latency(s)', 'hop', 'routed services', 'success rate', 'throughput',
             'com_utl', 'sto_utl', 'bandwidth_utl', 'cost']]
    for K in [3]:
        result_matrix = np.zeros(shape=(len(data[0]), repeat_times))
        for i in range(repeat_times):
            logging.info("It's running the {}th matrices in the {}th times.".format(K, i))
            graph = topology_obj.generate_topology()
            neighbors = topology_obj.get_neighbors(graph)
            bandwidth_matrix = topology_obj.get_bandwidth_matrix(graph)
            compute_matrix, storage_matrix = topology_obj.get_compute_and_storage_matrix(graph)
            # 初始化业务矩阵
            traffic_matrix_obj = traffic.TrafficMatrix(graph)
            traffic_matrices = traffic_matrix_obj.generate_traffic_matrices(K)
            num_edges = len(graph.edges)
            num_traffics = len([1 for tm in traffic_matrices for traffic in tm])
            dim = num_edges * num_traffics     # 变量数量

            # 实例化问题对象
            problem = problem_defination.MyProblem(G=graph,
                                                   num_edges=num_edges,
                                                   num_traffics=num_traffics,
                                                   traffic_matrices=traffic_matrices,
                                                   bandwidth_matrix=bandwidth_matrix,
                                                   compute_matrix=compute_matrix,
                                                   storage_matrix=storage_matrix,
                                                   neighbors=neighbors,
                                                   ips_per_gigabyte=ips_per_gigabyte)
            # 种群设置
            Encodings = ['P' for _ in range(num_traffics)]
            Fields = [ea.crtfld(Encodings[i],
                                problem.varTypes[i*num_edges:(i+1)*num_edges],
                                problem.ranges[:, i*num_edges:(i+1)*num_edges],
                                problem.borders[:, i*num_edges:(i+1)*num_edges])
                      for i in range(num_traffics)]
            population = ea.PsyPopulation(Encodings, Fields, NIND)
            # 算法设置
            algorithm = ea.moea_psy_NSGA2_templet(
                problem,
                population,
                MAXGEN=MAXGEN,     # 最大进化代数
                logTras=0,      # 表示每隔多少代记录一次日志信息，0表示不记录。
                verbose=True,
                drawing=False
            )

            try:
                # 求解
                [BestIndi, population] = algorithm.run()
                logging.info("{} - {} - The '{}'th of K='{}' get a solution!".format(__file__, __name__, i, K))
                # 保存结果
                solution = op.Solution()
                solution.init(traffic_matrix=traffic_matrices, graph=graph)
                solution.convert(BestIndi.ObjV, (0.4, 0.2, 0.2, 0.1, 0.1), BestIndi.Phen, problem)
                logging.info("{} - {} - The '{}'th of K='{}' start to get results!".format(__file__, __name__, i, K))
                res = op.Performance()
                result_matrix[0][i] = res.get_latency(solution, ips_per_gigabyte)   # latency(us)
                result_matrix[1][i] = res.get_hop(solution)                         # hop
                result_matrix[2][i] = res.get_routed_service(solution)              # routed services
                result_matrix[3][i] = res.get_success_rate(solution)                # success rate
                result_matrix[4][i] = res.get_throughput(solution)                  # throughput
                result_matrix[5][i] = res.get_compute_utilization(solution)         # com_utl
                result_matrix[6][i] = res.get_storage_utilization(solution)         # sto_utl
                result_matrix[7][i] = res.get_link_utilization(solution)            # bandwidth_utl
                result_matrix[8][i] = res.get_cost(solution)                        # cost
            except:
                logging.error("{} - {} - The '{}'th of K='{}' went wrong!".format(__file__, __name__, i, K))
                for j in range(len(data[0])):
                    result_matrix[j][i] = np.nan
        result_matrix = np.delete(result_matrix, np.where(np.isnan(result_matrix))[1], axis=1)
        data.append(('K={}'.format(K), np.average(result_matrix, axis=1)))
        print("{}K = {}{}".format('-'*50, K, '-'*50))
        print(pd.DataFrame(np.reshape(data[-1][1], newshape=(1, len(data[0]))), columns=data[0]))
    np.save('result.npy', data)
