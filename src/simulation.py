import logging
import scipy.io as scio
import matplotlib.pyplot as plt
import time

from input import *
from algorithms import Algorithms


def cal_res(adj_mat):
    data_utilization = []
    key_utilization = []
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            for t in adj_mat[i][j]:
                data_utilization.append((t.max_data_resource() - t.ava_data) / t.max_data_resource())
                key_utilization.append((t.max_key_resource() - t.ava_key) / t.max_key_resource())
    return np.mean(data_utilization), np.mean(key_utilization)


def cal_spf_res(adj_mat, max_resource):
    utilization = []
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            for t in adj_mat[i][j]:
                utilization.append((max_resource - t.resource) / max_resource)
    return np.mean(utilization)


if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini', )
    root = '../datasets/nsfnet/nsfnet.graphml'
    # file1 = 'mlf_suc.mat'
    # file2 = 'mlf_hop.mat'
    file3 = 'spf_lev.mat'
    nodes = 14
    tafs = [0.001, 0.005, 0.01, 0.05]
    lightpath_res = 100
    wavelength = 4
    successed_route = []
    MLF = [0, 0, 0, 1]
    SHF = [1, 0, 0, 0]
    HOP_ERR = [[], [], []]
    SUC_ERR = [[], [], []]
    LEV_ERR = [[], [], []]
    SUCCESS_ALLOC = []
    THP = []
    x = [(i+1)*0.05 for i in range(20)]
    for traffic_load in [(i+1)*0.05 for i in range(20)]:
        data_hop = []
        success = []
        levels = []
        ave_throughput = []
        for k in range(10):
            level = []
            traffic_res = int(lightpath_res * traffic_load)
            traffic_matrix = gen_traffic_matrix(nodes, tafs, max_res=traffic_res)
            G, adj_matrix_with_lightpath = gen_lightpath_topo(root, wavelengths=wavelength, tafs=tafs, max_res=lightpath_res)
            S = [[0 for j in range(nodes)] for i in range(nodes)]

            starttime = time.time()
            ave_hop = []
            for s in range(nodes):
                for d in range(nodes):
                    traffic = traffic_matrix[s][d]
                    # res = Algorithms.heuristic_algorithm(traffic, G, adj_matrix_with_lightpath, tafs, weight=SHF)
                    res = Algorithms.shorest_path_algorithm(traffic, adj_matrix_with_lightpath)
                    if res:
                        ave_hop.append(len(res))
                        S[s][d] = 1
                        level.append([tafs.index(adj_matrix_with_lightpath[a[0]][a[1]][a[2]].t) - tafs.index(traffic.t) for a in res])
                        # print(level)
            endtime = time.time()

            t = []
            for i in traffic_matrix:
                tmp = []
                for j in i:
                    if j:
                        tmp.append(j.resource)
                    else:
                        tmp.append(0)
                t.append(tmp)
            levels.append(np.mean([(np.sum([y**2 for y in x])/len(x))**(1/2) for x in level]))
            # data_hop.append(np.mean(ave_hop))
            # success.append(sum([sum(i) for i in S]) / (len(t)**2 - sum([sum([j == 0 for j in i]) for i in t])))
        print(np.mean(levels))
        LEV_ERR[0].append(np.min(levels))
        LEV_ERR[1].append(np.mean(levels))
        LEV_ERR[2].append(np.max(levels))
        # SUC_ERR[0].append(np.min(success))
        # SUC_ERR[1].append(np.mean(success))
        # SUC_ERR[2].append(np.max(success))
        # HOP_ERR[0].append(np.min(data_hop))
        # HOP_ERR[1].append(np.mean(data_hop))
        # HOP_ERR[2].append(np.max(data_hop))
    # scio.savemat(file1, {'min': SUC_ERR[0], 'mean': SUC_ERR[1], 'max': SUC_ERR[2]})
    # scio.savemat(file2, {'min': HOP_ERR[0], 'mean': HOP_ERR[1], 'max': HOP_ERR[2]})
    scio.savemat(file3, {'min': LEV_ERR[0], 'mean': LEV_ERR[1], 'max': LEV_ERR[2]})