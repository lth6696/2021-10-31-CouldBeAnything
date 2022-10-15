import copy
import logging

import numpy as np
import networkx as nx
import geatpy as ea

from result import output
from input.traffic import Traffic


class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self,
                 G,
                 num_edges,
                 num_traffics,
                 traffic_matrices,
                 bandwidth_matrix,
                 compute_matrix,
                 storage_matrix,
                 neighbors,
                 ips_per_gigabyte
                 ):
        self.__name = 'MultiObjPsyNSGA2'  # 初始化name（函数名称，可以随意设置）
        # 初始化内部属性
        self.G = G
        self.datacenter_nodes = [int(node) for node in self.G.nodes if self.G.nodes[node]['type'] == 'datacenter']
        self.num_edges = num_edges
        self.num_traffics = num_traffics
        self.traffic_matrices = traffic_matrices
        self.bandwidth_matrix = bandwidth_matrix
        self.compute_matrix = compute_matrix
        self.storage_matrix = storage_matrix
        self.neighbors = neighbors
        self.ips_per_gigabyte = ips_per_gigabyte
        self.oeo_conversion_latency = 10    # microsecond

        # 初始化模型参数
        Dim = num_edges * num_traffics  # 初始化Dim（决策变量维数）
        maxormins = [1, 1, -1, -1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [1] * Dim  # 决策变量下界
        ub = [self.num_edges] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        obj_nums = len(maxormins)
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            self.__name,
                            obj_nums,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    @staticmethod
    def _decode(source: int, destination: int, priority: np.array, neighbors: dict):
        """
        基于优先级的解码方法，将染色体的基因译码为可用路径。

        Lin, L., Gen, M. Priority-Based Genetic Algorithm for Shortest Path Routing Problem in OSPF. 2009. Intelligent
        and Evolutionary Systems. Studies in Computational Intelligence, vol 187. Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/978-3-540-95978-6_7

        :param source: int, 源节点ID
        :param destination: int, 目的节点ID
        :param priority: array, 优先级1-D数组
        :param neighbors: dict, 数据结构{节点：[节点邻居1，节点邻居2，...]}，记录各个节点邻居信息
        :return path: list, 解码后的路径
        """
        if source not in neighbors.keys() or destination not in neighbors.keys():
            logging.error("{} - {} - Source {} or destination {} not in the topology {}."
                          .format(__file__, __name__, source, destination, list(neighbors.keys())))
            return []
        path = [source]
        current_node = source
        priority[source] = 0        # 禁止路径折返源点
        while not path[-1] == destination:
            neighbor_nodes = neighbors[int(current_node)]
            if not neighbor_nodes:
                logging.error("{} - {} - Path from {} to {} gets trapped. Current path is {}"
                              .format(__file__, __name__, source, destination, path))
                break
            neighbor_nodes_priority = [priority[index] for index in neighbor_nodes]
            next_node = neighbor_nodes[np.argmax(neighbor_nodes_priority)]
            # 遍历过的节点优先级设置为0。
            if priority[next_node] != 0:
                path.append(next_node)
                priority[next_node] = 0
                current_node = next_node
            # 若后继均已遍历，则回退上一节点。
            else:
                # neighbors[int(current_node)] = []
                priority[next_node] = 0
                if len(path) <= 1:
                    break
                path.pop(-1)
                current_node = path[-1]
        return path

    @staticmethod
    def _des_priority_decode(source: int, destinations: set, priority: np.array, neighbors: dict):
        """
        基于优先级的解码方法，将染色体的基因译码为可用路径。与_decode方法不同，本方法的业务没有明确目的地，需要根据优先级确定。

        :param source: int, 源节点ID
        :param destinations: set, 所有目的节点的ID
        :param priority: array, 优先级1-D数组
        :param neighbors: dict, 数据结构{节点：[节点邻居1，节点邻居2，...]}，记录各个节点邻居信息
        :return path: list, 解码后的路径
        """
        if source not in neighbors.keys():
            logging.error("{} - {} - Source {} not in the topology {}."
                          .format(__file__, __name__, source, list(neighbors.keys())))
            return []
        path = [source]
        current_node = source
        priority[source] = 0  # 禁止路径折返源点
        while not path[-1] in destinations:
            neighbor_nodes = neighbors[int(current_node)]
            neighbor_nodes_priority = [priority[index] for index in neighbor_nodes]
            next_node = neighbor_nodes[np.argmax(neighbor_nodes_priority)]
            # 遍历过的节点优先级设置为0。
            if priority[next_node] != 0:
                path.append(next_node)
                priority[next_node] = 0
                current_node = next_node
            # 若后继均已遍历，则回退上一节点。
            else:
                priority[next_node] = 0
                if len(path) <= 1:
                    break
                path.pop(-1)
                current_node = path[-1]
        return path

    def _reserve_bandwidth(self, paths, bandwidth_matrix, traffic):
        """
        为业务沿路径预留带宽。
        若任一边缘带宽不足，则所有边缘的带宽都将不会被预留。
        :return: bool
        """
        if len(paths) < 1:
            raise ValueError
        (start, end) = paths.pop(0) # 此处源宿索引值等于节点ID
        # 若当前路径带宽不足，则阻塞业务
        if bandwidth_matrix[start][end] < traffic.req_bandwidth:
            return False
        # 若当前路径为最后一跳，则开始反向预留带宽
        if not paths:
            bandwidth_matrix[start][end] -= traffic.req_bandwidth
            return True
        # 若不为最后一跳，检查下一跳路径带宽
        else:
            # 从后往前依次预留带宽
            if self._reserve_bandwidth(paths, bandwidth_matrix, traffic):
                bandwidth_matrix[start][end] -= traffic.req_bandwidth
                return True
            # 若后继未能成功预留带宽
            else:
                return False

    def _get_routed_traffic(self, paths: dict):
        """
        该方法用于输出被成功路由业务的路径
        :param paths:
        :return:
        """
        bandwidth_matrix_ = copy.deepcopy(self.bandwidth_matrix)
        compute_matrix_ = copy.deepcopy(self.compute_matrix)
        storage_matrix_ = copy.deepcopy(self.storage_matrix)

        routed_paths = {}
        for key in paths.keys():
            k, col, dst = key  # 此处源宿索引值等于流量矩阵的行列数，不等于节点ID
            traffic = self.traffic_matrices[k][col]
            is_succeeded = True  # 标志位，判断业务是否被成功路由
            # 预留计算资源
            if compute_matrix_[dst] >= traffic.req_compute:
                compute_matrix_[dst] -= traffic.req_compute
            else:
                is_succeeded = False
            # 预留存储资源
            if storage_matrix_[dst] >= traffic.req_storage:
                storage_matrix_[dst] -= traffic.req_storage
            else:
                is_succeeded = False
            # 预留带宽
            if not self._reserve_bandwidth(list(zip(paths[key][:-1], paths[key][1:])), bandwidth_matrix_, traffic):
                is_succeeded = False
            # 满足时延需求
            traffic_latency = self._cal_latency(paths[key], traffic)
            if traffic_latency > traffic.req_latency:
                is_succeeded = False
            # 若带宽、计算资源、存储资源均有冗余，则业务可被认为成功路由
            if is_succeeded:
                routed_paths[key] = paths[key]
        return routed_paths

    def _eval_func(self, paths: dict):
        """
        本方法用于计算优化目标和约束条件。
        优化目标：
        1 - 最小化时延
        2 - 最小化成本
        3 - 最大路由业务数量
        4 - 最大平均算力利用率
        5 - 最大平均存储利用率
        约束条件：
        1 - 最大带宽
        2 - 最大
        :param paths:
        :return:
        """
        op = output.Output()            # 存储结果的类
        bandwidth_matrix_ = copy.deepcopy(self.bandwidth_matrix)
        compute_matrix_ = copy.deepcopy(self.compute_matrix)
        storage_matrix_ = copy.deepcopy(self.storage_matrix)

        succeed_traffic = np.zeros((len(self.neighbors.keys()), len(self.neighbors.keys())), dtype=int)
        hops = []
        latency = []
        cost = []
        for key in paths.keys():
            k, col, dst = key       # 此处源宿索引值等于流量矩阵的行列数，不等于节点ID
            traffic = self.traffic_matrices[k][col]
            is_succeeded = True    # 标志位，判断业务是否被成功路由
            # 预留计算资源
            if compute_matrix_[dst] >= traffic.req_compute:
                compute_matrix_[dst] -= traffic.req_compute
            else:
                is_succeeded = False
            # 预留存储资源
            if storage_matrix_[dst] >= traffic.req_storage:
                storage_matrix_[dst] -= traffic.req_storage
            else:
                is_succeeded = False
            # 预留带宽
            if not self._reserve_bandwidth(list(zip(paths[key][:-1], paths[key][1:])), bandwidth_matrix_, traffic):
                is_succeeded = False
            # 满足时延需求
            traffic_latency = self._cal_latency(paths[key], traffic)
            if traffic_latency > traffic.req_latency:
                is_succeeded = False
            # 若带宽、计算资源、存储资源均有冗余，则业务可被认为成功路由
            if is_succeeded:
                succeed_traffic[int(traffic.src)][dst] += 1
                hops.append(len(paths[key])-1)
                # 计算业务时延
                latency.append(traffic_latency)
                # 计算业务租用算力成本
                cost.append(np.sum([traffic.req_compute*self.G.nodes[str(dst)]['cost_compute'],
                                    traffic.req_storage*self.G.nodes[str(dst)]['cost_storage']]))
        computing_utilization = 1 - compute_matrix_ / self.compute_matrix
        storage_utilization = 1 - storage_matrix_ / self.storage_matrix
        op.eval_average_latency = np.average(latency)   # 单位second
        op.eval_average_cost = np.average(cost)
        op.eval_successfully_routed_services = np.sum(succeed_traffic)
        op.eval_average_hops = np.average(hops)
        op.eval_average_computing_utilization = np.average(computing_utilization[~np.isnan(computing_utilization)])
        op.eval_average_storage_utilization = np.average(storage_utilization[~np.isnan(storage_utilization)])
        return op

    def _cal_latency(self, path, traffic: Traffic):
        """
        本方法计算单条业务时延，业务时延包含四种：
        1 - 无线传输时延（s）
        2 - 有线传输时延（s）
        3 - 光电转换时延（s）
        4 - 计算时延（s）
        """
        transmission_latency = traffic.data * 8 / traffic.req_bandwidth  # second
        process_latency = traffic.data * self.ips_per_gigabyte / traffic.req_compute   # second
        # oeo_latency = (len(path) - 1) * self.oeo_conversion_latency * 1e-6
        return transmission_latency + process_latency

    def evalVars(self, Vars):
        # 初始化优化目标
        objV_min_average_latency = np.zeros((len(Vars), 1), dtype=float)
        objV_min_average_cost = np.zeros((len(Vars), 1), dtype=float)
        objV_max_successfully_routed_services = np.zeros((len(Vars), 1), dtype=int)
        objV_max_compute_utilization = np.zeros((len(Vars), 1), dtype=float)
        objV_max_storage_utilization = np.zeros((len(Vars), 1), dtype=float)
        # 初始化约束条件
        CV_bandwidth = np.zeros((len(Vars), 1), dtype=int)
        # 遍历种群每个个体，每个包含一条染色体。在网络模型内每条染色体的基因记录了优先级。
        current_traffic_index = 0
        for i, chromosome in enumerate(Vars):
            paths = {}  # 记录每条业务的路径，数据结构{(k, src, dst): [node1, node2, ...]}
            for k, traffic_matrix in enumerate(self.traffic_matrices):
                for col, traffic in enumerate(traffic_matrix):
                    if traffic.req_bandwidth == 0:
                        continue
                    priority = chromosome[current_traffic_index*self.num_edges: (current_traffic_index+1)*self.num_edges]
                    path = self._des_priority_decode(traffic.src, set(self.datacenter_nodes),
                                                     priority.copy(), self.neighbors)
                    paths[(k, col, path[-1])] = path
                    current_traffic_index += 1
            current_traffic_index = 0
            # 评价路径优劣
            output = self._eval_func(paths)
            # objV_min_average_hops[i] = output.eval_average_hops
            objV_min_average_latency[i] = output.eval_average_latency
            objV_min_average_cost[i] = output.eval_average_cost
            objV_max_successfully_routed_services[i] = output.eval_successfully_routed_services
            objV_max_compute_utilization[i] = output.eval_average_computing_utilization
            objV_max_storage_utilization[i] = output.eval_average_storage_utilization
        # CV = np.hstack([CV_bandwidth])
        ObjV = np.hstack([objV_min_average_latency,
                          objV_min_average_cost,
                          objV_max_successfully_routed_services,
                          objV_max_compute_utilization,
                          objV_max_storage_utilization])
        return ObjV

