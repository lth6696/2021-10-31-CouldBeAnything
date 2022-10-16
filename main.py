import logging.config

import pandas as pd

import result.analysis
import result.output as op
from input import network, traffic
from solver import problem_defination

import geatpy as ea
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')
    # 初始化网络拓扑
    repeat_times = 10
    ips_per_gigabyte = 1000
    topology_obj = network.Topology()
    graph = topology_obj.generate_topology()
    neighbors = topology_obj.get_neighbors(graph)
    bandwidth_matrix = topology_obj.get_bandwidth_matrix(graph)
    compute_matrix, storage_matrix = topology_obj.get_compute_and_storage_matrix(graph)
    # 初始化业务矩阵
    traffic_matrix_obj = traffic.TrafficMatrix(graph)
    data = [['latency(s)', 'hop', 'distance(km)',
             'routed services', 'success rate', 'throughput', 'com_utl', 'sto_utl', 'bandwidth_utl',
             'cost', 'ave_compute_req', 'ave_storage_req', 'ave_bandwidth_req']]
    for K in [1, 2, 3, 4, 5]:
        result_matrix = np.empty(shape=(len(data[0]), repeat_times))
        for i in range(repeat_times):
            logging.info("It's running the {}th matrices in the {}th times.".format(K, i))
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
            NIND = 50
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
                MAXGEN=100,     # 最大进化代数
                logTras=0,      # 表示每隔多少代记录一次日志信息，0表示不记录。
                verbose=True,
                drawing=False
            )

            # 求解
            [BestIndi, population] = algorithm.run()

            # 保存结果
            res = op.Result(graph, BestIndi.ObjV, BestIndi.Chroms, BestIndi.Phen)
            best_ObjV = res.get_best_ObjV()
            res.reserve_bandwdith(problem)
            result_matrix[0][i] = best_ObjV[0]                  # latency(us)
            result_matrix[1][i] = res.get_ave_hops(problem)     # hop
            result_matrix[2][i] = 0                             # distance(km)
            result_matrix[3][i] = best_ObjV[2]                  # routed services
            result_matrix[4][i] = best_ObjV[2]/num_traffics     # success rate
            result_matrix[5][i] = res.get_throughput(problem)   # throughput
            result_matrix[6][i] = best_ObjV[3]                  # com_utl
            result_matrix[7][i] = best_ObjV[4]                  # sto_utl
            result_matrix[8][i] = res.get_link_utilization(problem)     # bandwidth_utl
            result_matrix[9][i] = best_ObjV[1]                  # cost
            result_matrix[10][i] = res.get_ave_compute_req(problem)     # ave_compute_req
            result_matrix[11][i] = res.get_ave_storage_req(problem)     # ave_storage_req
            result_matrix[12][i] = res.get_ave_bandwidth_req(problem)   # ave_bandwidth_req
        data.append(('K={}'.format(K), np.average(result_matrix, axis=1)))
        print("{}K = {}{}".format('-'*50, K, '-'*50))
        print(pd.DataFrame(np.reshape(data[-1][1], newshape=(1, 13)), columns=data[0]))
    np.save('result.npy', data)
