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
    logging.config.fileConfig('logconfig.ini')
    root = '../datasets/nsfnet/nsfnet.graphml'
    file = 'spf_res.mat'
    nodes = 14
    tafs = [0.001, 0.005, 0.01, 0.05]
    lightpath_res = 100
    wavelength = 4
    successed_route = []
    SHF = [1, 0, 0, 0]
    MRU = [0, 0.5, 0.5, 0]
    HOP_ERR = [[], [], []]
    SUC_ERR = [[], [], []]
    DKR_ERR = [[], [], [], [], [], []]
    RES_ERR = [[], [], []]
    SUCCESS_ALLOC = []
    RUNTIME = []
    x = [(i+1)*0.05 for i in range(20)]
    for traffic_load in [(i+1)*0.05 for i in range(20)]:
        data_hop = []
        success = []
        resource = [[], []]
        resource_spf = []
        runtime = []
        for k in range(10):
            traffic_res = int(lightpath_res * traffic_load)
            traffic_matrix = gen_traffic_matrix(nodes, tafs, max_res=traffic_res)
            G, adj_matrix_with_lightpath = gen_lightpath_topo(root, wavelengths=wavelength, tafs=tafs, max_res=lightpath_res)
            S = [[0 for j in range(nodes)] for i in range(nodes)]

            starttime = time.time()
            ave_hop = []
            for s in range(nodes):
                for d in range(nodes):
                    traffic = traffic_matrix[s][d]
                    # res = Algorithms.heuristic_algorithm(traffic, G, adj_matrix_with_lightpath, tafs, weight=None)
                    res = Algorithms.shorest_path_algorithm(traffic, adj_matrix_with_lightpath)
                    if res:
                        ave_hop.append(len(res))
                        S[s][d] = 1
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
            # data_hop.append(np.mean(ave_hop))
            # success.append(sum([sum(i) for i in S]) / (len(t)**2 - sum([sum([j == 0 for j in i]) for i in t])))
            # r = cal_res(adj_matrix_with_lightpath)
            # resource[0].append(r[0])
            # resource[1].append(r[1])
            r = cal_spf_res(adj_matrix_with_lightpath, 100)
            resource_spf.append(r)
            # runtime.append((endtime - starttime) * 1000)
        # RUNTIME.append(np.mean(runtime))
        RES_ERR[0].append(np.min(resource_spf))
        RES_ERR[1].append(np.mean(resource_spf))
        RES_ERR[2].append(np.max(resource_spf))
        # DKR_ERR[0].append(np.min(resource[0]))
        # DKR_ERR[1].append(np.mean(resource[0]))
        # DKR_ERR[2].append(np.max(resource[0]))
        # DKR_ERR[3].append(np.min(resource[1]))
        # DKR_ERR[4].append(np.mean(resource[1]))
        # DKR_ERR[5].append(np.max(resource[1]))
        # SUC_ERR[0].append(np.min(success))
        # SUC_ERR[1].append(np.mean(success))
        # SUC_ERR[2].append(np.max(success))
        # HOP_ERR[0].append(np.min(data_hop))
        # HOP_ERR[1].append(np.mean(data_hop))
        # HOP_ERR[2].append(np.max(data_hop))
    # scio.savemat(file, {'runtime': RUNTIME})
    scio.savemat(file, {'min': RES_ERR[0], 'mean': RES_ERR[1], 'max': RES_ERR[2]})
    # scio.savemat(file, {'data_min': RES_ERR[0], 'data_mean': RES_ERR[1], 'data_max': RES_ERR[2],
    #                     'key_min': RES_ERR[3], 'key_mean': RES_ERR[4], 'key_max': RES_ERR[5]})
    # scio.savemat(file, {'min': SUC_ERR[0], 'mean': SUC_ERR[1], 'max': SUC_ERR[2]})
    # scio.savemat(file, {'min': HOP_ERR[0], 'mean': HOP_ERR[1], 'max': HOP_ERR[2]})