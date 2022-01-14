import networkx as nx
import random
import numpy as np
import pandas as pd
import datetime


pd.set_option('display.max_column', 200)
pd.set_option('display.width', 200)


class Traffic(object):
    def __init__(self, src, dst, data, t):
        self.src = src
        self.dst = dst
        self.data = data
        self.key = None
        self.t = t
        self.path = []

        self.__cal_key()

    def __cal_key(self):
        self.key = self.t * self.data


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


# generate traffic matrix
def gen_traffic_matrix(nodes, tafs):
    random.seed(0)
    traffic_matrix = [[None for _ in range(nodes)] for _ in range(nodes)]
    for src in range(nodes):
        for dst in range(nodes):
            if src != dst:
                traffic_matrix[src][dst] = Traffic(src,
                                                   dst,
                                                   random.randint(0, 10),
                                                   tafs[random.randint(0, len(tafs)-1)]
                                                   )
    return traffic_matrix


# generate lightpath topology
def gen_lightpath_topo(path, wavelengths, tafs):
    random.seed(0)
    G = nx.Graph(nx.read_graphml(path))

    adj_matrix = np.array(nx.adjacency_matrix(G).todense())
    adj_matrix_with_lightpath = [[[] for _ in range(len(G.nodes))] for _ in range(len(G.nodes))]
    for row in range(len(adj_matrix)):
        for col in range(len(adj_matrix)):
            adj_matrix[row][col] = adj_matrix[row][col] * random.randint(1, wavelengths)
            for t in range(adj_matrix[row][col]):
                lightpath = LightPath(row, col, t, 100, tafs[random.randint(0, len(tafs)-1)])
                adj_matrix_with_lightpath[row][col].append(lightpath)

    return adj_matrix, adj_matrix_with_lightpath


# A heuristic algorithm
def heuristic_algorithm():
    root = '../datasets/nsfnet/nsfnet.graphml'
    nodes = 14

    success_alloc_matrix = [[0 for _ in range(nodes)] for _ in range(nodes)]
    tafs = [0.001, 0.0001, 0.0005, 0.00001]
    traffic_matrix = gen_traffic_matrix(nodes, tafs)
    adj_matrix, adj_matrix_with_lightpath = gen_lightpath_topo(root, wavelengths=4, tafs=tafs)

    for src in range(len(traffic_matrix)):
        for dst in range(len(traffic_matrix)):
            traffic = traffic_matrix[src][dst]
            # 若流量矩阵从src至dst存在连接请求，为其路由并分配资源
            if traffic:
                # 路由
                paths = function_multi_hop(4, src, traffic, adj_matrix_with_lightpath)
                # 若路由成功，使用success_alloc_matrix进行记录
                if paths:
                    success_alloc_matrix[src][dst] = 1
                # 分配资源
                for path in paths:
                    adj_matrix_with_lightpath[path[0]][path[1]][path[2]].ava_data -= traffic.data
                    adj_matrix_with_lightpath[path[0]][path[1]][path[2]].ava_key -= traffic.key

    return success_alloc_matrix


# 基于多约束的最短路径算法，约束有：
# 1 - 安全等级
# 2 - 资源多寡
def function_multi_hop(max_hop, src, traffic, adj_matrix_with_lightpath):
    dst = traffic.dst

    # 设定迭代深度
    if max_hop <= 0:
        return []

    # 判断本次目的地是否可达
    for lightpath in adj_matrix_with_lightpath[src][dst]:
        if lightpath.t >= traffic.t and \
                lightpath.ava_data >= traffic.data and \
                lightpath.ava_key >= traffic.data * traffic.t:
            path = (src, dst, lightpath.index)
            return [path]

    # 若目的地不可直达
    tmp_path = []
    path_len = 10
    for j in range(len(adj_matrix_with_lightpath[src])):
        if j != dst:
            for inter_lp in adj_matrix_with_lightpath[src][j]:
                # 三个约束条件，光路等级高于业务，光路有可用通资源，光路有可用密资源
                if inter_lp.t >= traffic.t and \
                        inter_lp.ava_data >= traffic.data and \
                        inter_lp.ava_key >= traffic.data * traffic.t:
                    paths = function_multi_hop(max_hop-1, j, traffic, adj_matrix_with_lightpath)
                    if len(paths) < path_len and len(paths) > 0:
                        paths.insert(0, (src, j, inter_lp.index))
                        path_len = len(paths)
                        tmp_path = paths
    return tmp_path


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    success_alloc_matrix = heuristic_algorithm()
    endtime = datetime.datetime.now()
    print((endtime - starttime))
    print(sum([sum(i) for i in success_alloc_matrix]))
