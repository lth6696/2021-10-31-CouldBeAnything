import logging

import numpy as np

from solver import problem_defination as prod


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


class Solution:
    """
    该类用于存储算法结果，包含NSGA和Heuristic算法。
    """
    def __init__(self):
        # 涉及性能分析的结果包括流量矩阵、拓扑、路由路径
        self.traffic_matrix = None
        self.graph = None
        self.path = None

    def init(self, **kwargs):
        """
        本方法用于启发式算法直接初始化
        """
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                raise ValueError('{} does not have the attribution of {}.'.format(__name__, key))

    def convert(self, objv, weight, phen, problem: prod.MyProblem):
        """
        本方法将遗传算法结果转换为通用结果
        self.get_best_phen()
        self._get_paths(problem)
        self._get_routed_traffic(self.paths, problem)
        """
        if self.graph is None and self.traffic_matrix is None:
            raise ValueError
        # 获取最佳解变量
        if len(objv) == 1:
            best_solution = phen[0]
        elif len(objv) > 1:
            min_ = np.min(objv, axis=0)
            max_ = np.max(objv, axis=0)
            norm_ObjV = (objv - min_) / (max_ - min_)
            norm_ObjV[np.isinf(norm_ObjV)] = 1
            norm_ObjV[np.isnan(norm_ObjV)] = 0
            norm_ObjV[:, :2] = 1 - norm_ObjV[:, :2]     # 前两列为最小化目标
            best_objv_index = np.argmax(norm_ObjV.dot(np.array(weight)))       # 根据权重选择最优解
            best_solution = phen[best_objv_index]
        else:
            return None
        # 路径译码
        chrom_len = len(self.graph.edges)
        decode_paths = {}
        for k, tm in enumerate(self.traffic_matrix):
            for col, traffic in enumerate(tm):
                chrom_index = (k * len(tm)) + col
                path = problem._des_priority_decode(traffic.src,
                                                    problem.datacenter_nodes,
                                                    best_solution[chrom_index * chrom_len: (chrom_index + 1) * chrom_len].copy(),
                                                    problem.neighbors)
                decode_paths[(k, col, path[-1])] = path
        # 判断业务成功路由
        self.path = problem._get_routed_traffic(decode_paths)
        # 预留带宽
        for key in self.path:
            traffic = self.traffic_matrix[key[0]][key[1]]
            for (u, v) in zip(self.path[key][:-1], self.path[key][1:]):
                self.graph[str(u)][str(v)]['bandwidth'] -= traffic.req_bandwidth
            self.graph.nodes[str(key[-1])]['compute'] -= traffic.req_compute
            self.graph.nodes[str(key[-1])]['storage'] -= traffic.req_storage
        return self.path


class Performance:
    """
    获取各项性能指标
    """
    @staticmethod
    def get_latency(solution: Solution, ips_per_gigabyte: int):
        latency = []
        for (k, col, dst) in solution.path:
            traffic = solution.traffic_matrix[k][col]
            trans_latency = traffic.data * 8 / traffic.req_bandwidth  # second
            proce_latency = traffic.data * ips_per_gigabyte / traffic.req_compute   # second
            latency.append(trans_latency+proce_latency)
        return np.mean(latency)

    @staticmethod
    def get_hop(solution: Solution):
        hop = []
        for key in solution.path:
            hop.append(len(solution.path[key]) - 1)
        return np.mean(hop)

    @staticmethod
    def get_success_rate(solution: Solution):
        K = len(solution.traffic_matrix)
        cols = len(solution.traffic_matrix[0])
        success_services = len(solution.path)
        total_services = K * cols
        return success_services / total_services * 100

    @staticmethod
    def get_throughput(solution: Solution):
        throughput = 0
        for (k, col, dst) in solution.path:
            traffic = solution.traffic_matrix[k][col]
            throughput += traffic.req_bandwidth
        return throughput

    @staticmethod
    def get_link_utilization(solution: Solution):
        link_utilization = []
        for (u, v) in solution.graph.edges:
            link_utilization.append(1 - solution.graph[u][v]['bandwidth'] / solution.graph[u][v]['max_bandwidth'])
        return np.mean(link_utilization) * 100

    @staticmethod
    def get_storage_utilization(solution: Solution):
        storage_utilization = []
        for node in solution.graph.nodes:
            if solution.graph.nodes[node]['type'] == 'datacenter':
                storage_utilization.append(1 - solution.graph.nodes[node]['storage'] / solution.graph.nodes[node]['max_storage'])
        return np.mean(storage_utilization) * 100

    @staticmethod
    def get_compute_utilization(solution: Solution):
        compute_utilization = []
        for node in solution.graph.nodes:
            if solution.graph.nodes[node]['type'] == 'datacenter':
                compute_utilization.append(1 - solution.graph.nodes[node]['compute'] / solution.graph.nodes[node]['max_compute'])
        return np.mean(compute_utilization) * 100

    @staticmethod
    def get_cost(solution: Solution):
        cost = []
        for (k, col, dst) in solution.path:
            traffic = solution.traffic_matrix[k][col]
            cost.append(np.sum([traffic.req_compute * solution.graph.nodes[str(dst)]['cost_compute'],
                                traffic.req_storage * solution.graph.nodes[str(dst)]['cost_storage']]))
        return np.mean(cost)

    @staticmethod
    def get_routed_service(solution: Solution):
        return len(solution.path)