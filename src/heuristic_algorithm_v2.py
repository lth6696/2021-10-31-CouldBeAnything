import logging
import scipy.io as scio
import matplotlib.pyplot as plt
import time
import random
import networkx as nx
import numpy as np
import pandas as pd
import logging
import logging.config


class Traffic(object):
    def __init__(self, src, dst, resource, t):
        self.src = src
        self.dst = dst
        self.resource = resource
        self.data = None
        self.key = None
        self.t = t
        self.path = []

        self.__cal_resource()

    def __cal_resource(self):
        self.data = self.resource * 1 / (self.t + 1)
        self.key = self.resource * self.t / (self.t + 1)


class LightPath(object):
    def __init__(self, start, end, index, resource, t):
        self.start = start
        self.end = end
        self.index = index
        self.resource = resource
        self.ava_data = 0
        self.ava_key = 0
        self.t = t

        self.__cal_ava_resource()

    def __cal_ava_resource(self):
        self.ava_data = self.resource * 1 / (self.t + 1)
        self.ava_key = self.resource * self.t / (self.t + 1)

    def max_data_resource(self):
        return self.resource * 1 / (self.t + 1)

    def max_key_resource(self):
        return self.resource * self.t / (self.t + 1)


# generate traffic matrix
def gen_traffic_matrix(nodes, tafs, max_res):
    random.seed(1)
    traffic_matrix = [[None for _ in range(nodes)] for _ in range(nodes)]
    for src in range(nodes):
        for dst in range(nodes):
            resource = np.abs(np.random.normal(loc=1, scale=0.4)) * max_res         # traffic load
            if src != dst and resource != 0:
                traffic_matrix[src][dst] = Traffic(src,
                                                   dst,
                                                   resource,
                                                   tafs[random.randint(0, len(tafs)-1)]
                                                   )
    return traffic_matrix


# generate lightpath topology
def gen_lightpath_topo(path, wavelengths, tafs, max_res):
    random.seed(1)
    G = nx.Graph(nx.read_graphml(path))
    adj_matrix = np.array(nx.adjacency_matrix(G).todense())
    adj_matrix_with_lightpath = [[[] for _ in range(len(G.nodes))] for _ in range(len(G.nodes))]
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix)):
            adj_matrix[row][col] = adj_matrix[row][col] * random.randint(1, wavelengths)
            for t in range(adj_matrix[row][col]):
                t_index = len(tafs) - 1 - t % len(tafs)
                # t_index = random.randint(0, len(tafs)-1)
                logging.info("Lightpath {}-{}-{} level {}".format(row, col, t, tafs[t_index]))
                lightpath = LightPath(row, col, t, max_res, tafs[t_index])
                adj_matrix_with_lightpath[row][col].append(lightpath)
    logging.info("Adj Matrix\n{}".format(pd.DataFrame(adj_matrix)))
    return G, adj_matrix_with_lightpath


def heuristic_algorithm(traffic, Graph, AdjMat, tafs, weight=None, max_k=4, cutoff=None):
    """
    step 0 - initialize
    step 1 - check if there exists the fxxking traffic
    step 2 - based on k-shortest paths, route the traffic
    step 3 - based on First Fit, allocate the lightpath for each hop
    step 4 - calculate the weight of different paths, select the biggest one
    step 5 - return PATH
    """
    # step 1
    if weight is None:
        weight = [0.25, 0.25, 0.25, 0.25]
    if traffic:
        s = traffic.src     # Int
        d = traffic.dst
        logging.info("Current Traffic {}-{} request {} data and {} key with {} level".format(traffic.src, traffic.dst,
                                                                                             traffic.data, traffic.key,
                                                                                             traffic.t))
    else:
        return False

    # step 0
    random.seed(123)
    tl = traffic.t      # traffic level
    paths_with_index = []
    denominator = max(tafs) - min(tafs)

    # step 2
    paths = list(nx.shortest_simple_paths(Graph, str(s), str(d)))    # k-shortest paths
    # logging.info("{}-{} all available paths \n{}".format(s, d, paths))
    # dividend = 4
    # quotient = math.ceil(len(paths) / dividend)
    # remainder = len(paths) % dividend if len(paths) % dividend != 0 else dividend
    # paths = [paths[i+4*j] for i in range(remainder) for j in range(quotient)]
    # logging.info("{}-{} selected available paths \n{}".format(s, d, paths))
    # step 3
    for path in paths:
        if len(paths_with_index) >= max_k:
            break
        path_with_index = []
        for i in range(len(path)-1):
            m = int(path[i])
            n = int(path[i+1])
            multi_edges = []
            for l in AdjMat[m][n]:
                # logging.info("Check LP {}-{}-{} level:{} data:{} key:{}".format(l.start, l.end, l.index, l.t, l.ava_data, l.ava_key))
                if l.t < tl or l.ava_data < traffic.data or l.ava_key < traffic.key:
                    # logging.info("The LP {}-{}-{} does not meet level {} data {} key {} needs.".format(l.start, l.end, l.index, l.t, l.ava_data, l.ava_key))
                    continue
                multi_edges.append((l.index, l.t))
            multi_edges.sort(key=lambda s: s[1], reverse=False)             # 按升序排序
            if not multi_edges:
                path_with_index = []
                break
            path_with_index.append((m, n, multi_edges[0][0]))                   # First Fit
        if path_with_index:
            paths_with_index.append(path_with_index)
    # logging.info("Traffic {}-{} can use LP \n{}".format(s, d, pd.DataFrame(paths_with_index)))

    # step 4
    if not paths_with_index:
        return False
    path_len = [len(path) for path in paths_with_index]
    max_hop = max(path_len)
    a0 = [i/max_hop for i in path_len]        # the length of hop
    a1, a2, a3 = [], [], []                 # a1-data utilization a2-key utilization a3-level
    for path in paths_with_index:
        D, K, L = 0, 0, 0
        for hop in path:
            lightpath = AdjMat[hop[0]][hop[1]][hop[2]]
            D += (lightpath.max_data_resource() - lightpath.ava_data) / lightpath.max_data_resource()
            K += (lightpath.max_key_resource() - lightpath.ava_key) / lightpath.max_key_resource()
            L += ((lightpath.t - traffic.t) / denominator) ** 2
        a1.append(D / len(path))
        a2.append(K / len(path))
        a3.append((L / len(path))**(1/2))
    if len(a0) == len(a1) and len(a1) == len(a2) and len(a2) == len(a3):
        score = np.dot(weight, np.vstack((a0, a1, a2, a3)))
        index = list(score).index(min(score))
    else:
        return False
    # print(np.vstack((a0, a1, a2, a3)))
    logging.info("{}-{} selected path: {}".format(s, d, paths_with_index[index]))

    # step 5
    for hop in paths_with_index[index]:
        lp = adj_matrix_with_lightpath[hop[0]][hop[1]][hop[2]]
        lp.ava_data -= traffic.data
        lp.ava_key -= traffic.key

    return paths_with_index[index]


def cal_res(adj_mat):
    data_utilization = []
    key_utilization = []
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            for t in adj_mat[i][j]:
                data_utilization.append((t.max_data_resource() - t.ava_data) / t.max_data_resource())
                key_utilization.append((t.max_key_resource() - t.ava_key) / t.max_key_resource())
    return np.mean(data_utilization), np.mean(key_utilization)

if __name__ == '__main__':
    logging.config.fileConfig('logconfig.ini')
    root = '../datasets/nsfnet/nsfnet.graphml'
    file = 'runtime.mat'
    nodes = 14
    tafs = [0.001, 0.005, 0.01, 0.05]
    lightpath_res = 100
    wavelength = 4
    successed_route = []
    SHF = [1, 0, 0, 0]
    MRU = [0, 0.5, 0.5, 0]
    HOP_ERR = [[], [], []]
    SUC_ERR = [[], [], []]
    RES_ERR = [[], [], [], [], [], []]
    SUCCESS_ALLOC = []
    RUNTIME = []
    x = [(i+1)*0.05 for i in range(20)]
    for traffic_load in [0.4]:
        data_hop = []
        success = []
        resource = [[], []]
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
                    res = heuristic_algorithm(traffic, G, adj_matrix_with_lightpath, tafs, weight=None)
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
            runtime.append((endtime - starttime) * 1000)
            print(runtime)
        RUNTIME.append(np.mean(runtime))
        # RES_ERR[0].append(np.min(resource[0]))
        # RES_ERR[1].append(np.mean(resource[0]))
        # RES_ERR[2].append(np.max(resource[0]))
        # RES_ERR[3].append(np.min(resource[1]))
        # RES_ERR[4].append(np.mean(resource[1]))
        # RES_ERR[5].append(np.max(resource[1]))
        # SUC_ERR[0].append(np.min(success))
        # SUC_ERR[1].append(np.mean(success))
        # SUC_ERR[2].append(np.max(success))
        # HOP_ERR[0].append(np.min(data_hop))
        # HOP_ERR[1].append(np.mean(data_hop))
        # HOP_ERR[2].append(np.max(data_hop))
    scio.savemat(file, {'runtime': RUNTIME})
    # scio.savemat(file, {'data_min': RES_ERR[0], 'data_mean': RES_ERR[1], 'data_max': RES_ERR[2],
    #                     'key_min': RES_ERR[3], 'key_mean': RES_ERR[4], 'key_max': RES_ERR[5]})
    # scio.savemat(file, {'min': SUC_ERR[0], 'mean': SUC_ERR[1], 'max': SUC_ERR[2]})
    # scio.savemat(file, {'min': HOP_ERR[0], 'mean': HOP_ERR[1], 'max': HOP_ERR[2]})