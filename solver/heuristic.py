import networkx as nx
import numpy as np

from input.traffic import Traffic


class ComputingAwareRoutingAllocating:
    """
    本类实现基于算力感知路由分配算法[1]。
    [1] 孙钰坤,张兴,雷波. 边缘算力网络中智能算力感知路由分配策略研究[J]. 无线电通信技术,2022,48(1):60-67.
    """
    def run(
            self,
            graph: nx.DiGraph,
            traffic_matrix: np.ndarray,
    ):
        """
        算法运行入口。
        :param graph: nx.DiGraph, 拓扑图
        :param traffic_matrix: np.ndarray, 流量矩阵
        :return path: dict, 所有成功路由的业务路径
        """
        path = {}
        computing_nodes = [node for node in graph.nodes if graph.nodes[node]['type'] == 'datacenter']
        for k, tm in enumerate(traffic_matrix):
            for col, traffic in enumerate(tm):
                paths_to_computing_nodes = self._cal_path(str(traffic.src), computing_nodes, graph)
                selected_path = self._sel_path(paths_to_computing_nodes, graph, traffic)
                if selected_path:
                    path[(k, col, selected_path[-1])] = selected_path
        return path

    def _cal_path(self, client_node: str, computing_nodes: list, graph: nx.DiGraph):
        """
        本方法计算一条最短路径
        :return path: list, 最短路径
        """
        paths_to_computing_nodes = []
        for dst in computing_nodes:
            self._insert_path(nx.shortest_path(graph, client_node, dst, weight='cost'), paths_to_computing_nodes)
        return paths_to_computing_nodes

    def _sel_path(self, paths: list, graph: nx.DiGraph, traffic: Traffic):
        while paths:
            current_path = paths.pop(0)
            if self._reserve_resource(current_path, graph, traffic):
                return current_path
        return []

    def _reserve_resource(self, path: list, graph: nx.DiGraph, traffic: Traffic, ips_per_gigabyte: int = 1000):
        # check latency requirement
        transmission_latency = traffic.data * 8 / traffic.req_bandwidth  # second
        process_latency = traffic.data * ips_per_gigabyte / traffic.req_compute   # second
        if transmission_latency + process_latency > traffic.req_latency:
            return False
        # check computing and storage resource
        if not (graph.nodes[path[-1]]['compute'] >= traffic.req_compute and graph.nodes[path[-1]]['storage'] >= traffic.req_storage):
            return False
        # reserve bandwidth
        if not self._reserve_bandwidth(graph, list(zip(path[:-1], path[1:])), traffic):
            return False
        # reserve computing and storage resource
        graph.nodes[path[-1]]['compute'] -= traffic.req_compute
        graph.nodes[path[-1]]['storage'] -= traffic.req_storage
        return True

    def _reserve_bandwidth(self, graph: nx.DiGraph, path_seg: list, traffic: Traffic):
        """
        本方法从前向后检查带宽多寡，从后向前预留带宽
        :param graph: nx.DiGraph, 有向图
        :param path_seg: list, 路径分段集合，其包含多组(u, v)对
        :param traffic: obj, 流量对象
        :return: bool, 是否成功预留带宽
        """
        (u, v) = path_seg.pop(0)
        # 判断路径带宽是否足够
        if graph[u][v]['bandwidth'] < traffic.req_bandwidth:
            return False
        # 若当前为最后一跳，即路径分段为空集合
        if not path_seg:
            graph[u][v]['bandwidth'] -= traffic.req_bandwidth
            return True
        # 若不为最后一跳，则向后递归
        else:
            # 若后继成功预留带宽，则继续向前预留
            if self._reserve_bandwidth(graph, path_seg, traffic):
                graph[u][v]['bandwidth'] -= traffic.req_bandwidth
                return True
            # 若后继带宽不足，则不会预留带宽
            else:
                return False

    @ staticmethod
    def _insert_path(x: list, l: list):
        """
        本方法用于将路径按长度插入路径表中，核心采用二分插入算法
        :param x: list, 路径
        :param l: list, 路径表，默认有序
        :return: l
        """
        if not x:
            return l
        if not l:
            l.append(x)
            return l
        length = len(l)
        left = 0
        right = length - 1
        # 判断边界
        if len(x) <= len(l[left]):
            l.insert(left, x)
            return l
        elif len(x) >= len(l[right]):
            l.append(x)
            return l
        else:
            pass
        # 二分查找并插入
        while left <= right:
            mid = int(left + (right - left) / 2)
            if len(l[mid]) == len(x):
                l.insert(mid, x)
            elif len(l[mid]) < len(x):
                if len(l[mid+1]) > len(x):
                    l.insert(mid+1, x)
                else:
                    left = mid + 1
            elif len(l[mid]) > len(x):
                if len(l[mid+1]) < len(x):
                    l.insert(mid, x)
                else:
                    right = mid - 1
            else:
                pass
        return l
