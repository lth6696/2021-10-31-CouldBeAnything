import logging

import numpy as np

from solver import problem_defination as pd


class Output:
    """
    该类用于保存对算法输出结果的评价值，可能存在有两类：
    1 - 优化目标值
    2 - 约束条件值
    """
    def __init__(self):
        # 优化目标
        self.eval_successfully_routed_services = 0
        self.eval_average_hops = 0
        self.eval_average_computing_utilization = 0
        self.eval_average_storage_utilization = 0
        self.eval_average_latency = 0
        self.eval_average_cost = 0
        # 约束条件
        self.cv_oversubscribe_bandwidth_edges = 0


class Result:
    """
    该类用于保存算法输出的最优化方案
    """
    def __init__(self, graph, ObjV, chrom, phen, weight=None):
        self.graph = graph
        # 算法结果
        self.optimal_ObjV = ObjV
        self.optimal_chroms = chrom
        self.optimal_phen = phen
        self.weight = weight if weight else (0.4, 0.2, 0.2, 0.1, 0.1)
        self.paths = None
        self.routed_paths = None
        # 最佳结果
        self.best_ObjV = None
        self.best_ObjV_index = None
        self.best_phen = None

        self.is_reserved = False

    def _get_best_ObjV_index(self):
        # 归一化
        min_ = np.min(self.optimal_ObjV, axis=0)
        max_ = np.max(self.optimal_ObjV, axis=0)
        normalized_ObjV = (self.optimal_ObjV - min_) / (max_ - min_)
        normalized_ObjV[~np.isfinite(normalized_ObjV)] = 0
        normalized_ObjV[:, :2] = 1 - normalized_ObjV[:, :2]
        # 根据权重选择最优解
        vars = normalized_ObjV.dot(np.array(self.weight))
        max_vars_index = np.where(vars == np.max(vars))
        if len(max_vars_index) > 1:
            logging.warning("{} - {} - There are {} best phens.".format(__file__, __name__, len(max_vars_index)))
        self.best_ObjV_index = int(max_vars_index[0])
        return self.best_ObjV_index

    def _get_paths(self, problem):
        if self.paths is not None:
            return self.paths
        # 变量长度为num_edges*num_traffic
        chrom_index = 0
        chrom_len = problem.num_edges
        paths = {}
        for k, tm in enumerate(problem.traffic_matrices):
            for col, traffic in enumerate(tm):
                path = problem._des_priority_decode(traffic.src,
                                                    problem.datacenter_nodes,
                                                    self.best_phen[chrom_index * chrom_len: (chrom_index + 1) * chrom_len].copy(),
                                                    problem.neighbors)
                paths[(k, col, path[-1])] = path
                chrom_index += 1
        self.paths = paths
        return self.paths

    def _get_routed_traffic(self, paths: dict, problem: pd.MyProblem):
        if self.routed_paths is None:
            self.routed_paths = problem._get_routed_traffic(paths)
        return self.routed_paths

    def get_best_ObjV(self):
        if not self.best_ObjV_index:
            self._get_best_ObjV_index()
        self.best_ObjV = self.optimal_ObjV[self.best_ObjV_index]
        return self.best_ObjV

    def get_best_phen(self):
        # 若尚未选出最佳解的索引
        if not self.best_ObjV_index:
            self._get_best_ObjV_index()
        # 若尚未给出最佳解对应的变量
        if self.best_phen is None:
            self.best_phen = self.optimal_phen[self.best_ObjV_index]
        return self.best_phen

    def get_ave_hops(self, problem: pd.MyProblem):
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        ave_hop = np.average([len(self.routed_paths[index])-1 for index in self.routed_paths])
        return ave_hop

    def get_throughput(self, problem: pd.MyProblem):
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        throughput = [problem.traffic_matrices[k][col].req_bandwidth for (k, col, dst) in self.routed_paths]
        return np.sum(throughput)

    def get_ave_bandwidth_req(self, problem: pd.MyProblem):
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        ave_bandwidth_req = [problem.traffic_matrices[k][col].req_bandwidth for (k, col, dst) in self.routed_paths]
        return np.average(ave_bandwidth_req)

    def get_ave_compute_req(self, problem: pd.MyProblem):
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        ave_compute_req = [problem.traffic_matrices[k][col].req_compute for (k, col, dst) in self.routed_paths]
        return np.average(ave_compute_req)

    def get_ave_storage_req(self, problem: pd.MyProblem):
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        ave_storage_req = [problem.traffic_matrices[k][col].req_storage for (k, col, dst) in self.routed_paths]
        return np.average(ave_storage_req)

    def get_link_utilization(self, problem: pd.MyProblem):
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        link_utilization = []
        for (u, v) in self.graph.edges:
            link_utilization.append(1-self.graph[u][v]['bandwidth']/self.graph[u][v]['max_bandwidth'])
        return np.mean(link_utilization)

    def reserve_bandwdith(self, problem):
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        for key in self.routed_paths:
            (k, col, dst) = key
            for (u, v) in zip(self.routed_paths[key][:-1], self.routed_paths[key][1:]):
                traffic = problem.traffic_matrices[k][col]
                self.graph[str(u)][str(v)]['bandwidth'] -= traffic.req_bandwidth