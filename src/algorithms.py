
import random
import networkx as nx
import numpy as np
import logging
import logging.config


class Algorithms:
    @staticmethod
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
                # D += (lightpath.max_data_resource() - lightpath.ava_data) / lightpath.max_data_resource()
                D += (lightpath.ava_data) / lightpath.max_data_resource()   # max resource utilization
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
            lp = AdjMat[hop[0]][hop[1]][hop[2]]
            lp.ava_data -= traffic.data
            lp.ava_key -= traffic.key

        return paths_with_index[index]

    @staticmethod
    def shorest_path_algorithm(traffic, AdjMat):
        """
        step 0 - initialize MultiDiGraph
        step 1 - check if there exists traffic
        step 2 - route traffic based on SPF (cost)
        step 3 - allocate light path
        step 4 - return PATH
        """
        # step 0
        G = nx.MultiDiGraph()
        G.add_nodes_from([i for i in range(len(AdjMat))])
        for s in range(len(AdjMat)):
            for d in range(len(AdjMat)):
                for l in AdjMat[s][d]:
                    G.add_edge(s, d, index=l.index, resource=l.resource, cost=1/l.resource)

        # step 1
        if traffic:
            s = traffic.src
            d = traffic.dst
            logging.info("Current Traffic {}-{}".format(s, d))
        else:
            return False

        # step 2
        path = nx.shortest_path(G, s, d, weight='cost')

        # step 3
        path_with_index = []
        for i in range(len(path) - 1):
            m = int(path[i])
            n = int(path[i + 1])
            lp_cost = [G[m][n][t]['cost'] for t in range(len(G[m][n]))]
            min_cost_lightpath = -1
            while True:
                if len(lp_cost) == 0:
                    return False
                min_cost_lightpath = lp_cost.index(min(lp_cost))
                if G[m][n][min_cost_lightpath]['resource'] >= traffic.resource:
                    break
                else:
                    del lp_cost[min_cost_lightpath]
            if min_cost_lightpath == -1:
                path_with_index.clear()
                return False
            path_with_index.append((m, n, min_cost_lightpath))  # First Fit

        # step 4
        for hop in path_with_index:
            lp = AdjMat[hop[0]][hop[1]][hop[2]]
            lp.resource -= traffic.resource

        return path_with_index




