import networkx as nx
import numpy as np
from collections import defaultdict


class Result(object):
    def __init__(self):
        self.num_traffic_success_mapping = 0
        self.num_traffic_fail_mapping = 0
        self.num_traffic = 0
        self.traffic_mapping_success_rate = 0
        self.set_block_traffic = {}
        self.set_success_traffic = {}

        self.traffic_matrix = None
        self.MultiDiG = None

    def set_attrs(self, nsuc: int, nfail: int, nt: int, setblock: dict, setsuccess: dict, traffic_matrix: list, MultiDiG: nx.MultiDiGraph):
        """
        :param nsuc: int, the number of traffic that success to map.
        :param nfail: int, the number of traffic that fail to map.
        :param nt: int, the number of traffic.
        :param setblock: dict, the set of blocked traffic.
        :param setsuccess: dict, the set of success-map traffic
        """
        self.num_traffic = nt
        self.num_traffic_success_mapping = nsuc
        self.num_traffic_fail_mapping = nfail
        self.set_block_traffic = setblock
        self.set_success_traffic = setsuccess
        self.traffic_mapping_success_rate = self._get_success_mapping_rate()
        self.traffic_matrix = traffic_matrix
        self.MultiDiG = MultiDiG

        self._verify_quantity_correct()

    def _get_success_mapping_rate(self):
        if self.num_traffic == 0:
            raise Exception('The number of traffic can not be zero.')
        return self.num_traffic_success_mapping / self.num_traffic * 100

    def _verify_quantity_correct(self):
        if self.num_traffic != self.num_traffic_success_mapping + self.num_traffic_fail_mapping:
            raise Exception('Quantity relationship is wrong.')


class ResultAnalysisImpl(object):
    def __init__(self, result: Result):
        self.result = result

    def analysis_traffic_block_rate_under_different_situation(self, result: Result):
        """
        analysis_result_to_matrix:
                situation1  situation2  ...
        level1        0.0%        0.0%  ...
        level2        0.0%        0.0%  ...
           ...         ...         ...  ...
        """
        analysis_result = {}    # {level1: {'0x01': int, '0x02': int, ...}, ...}
        for key in result.set_block_traffic.keys():
            analysis_result[key] = defaultdict(int)
            for traffic in result.set_block_traffic[key]:
                analysis_result[key][traffic['block_reason']] += 1
        # situations = set()
        # for key in analysis_result:
        #     situations |= set(analysis_result[key].keys())
        # situations = sorted(situations)
        situations = ['0x01', '0x02']
        analysis_result_to_matrix = [[] for _ in range(len(analysis_result.keys()))]
        for i, level in enumerate(sorted(analysis_result.keys())):
            analysis_result_to_matrix[i].append(level)
            for situation in situations:
                if situation in analysis_result[level].keys():
                    analysis_result_to_matrix[i].append(
                        analysis_result[level][situation] / result.num_traffic * 100
                    )
                else:
                    analysis_result_to_matrix[i].append(0.0)
        return analysis_result_to_matrix

    def analyze_level_distribution_for_lightpath(self, MultiDiG: nx.MultiDiGraph):
        analysis_result = defaultdict(int)
        levels = set()
        for (ori, sin, index) in MultiDiG.edges:
            level = MultiDiG[ori][sin][index]['level']
            levels |= {level}
            analysis_result[level] += 1
        levels = sorted(levels)
        analysis_result_to_matrix = [analysis_result[key] for key in levels]
        return analysis_result_to_matrix

    def analyze_throughput_for_each_level(self, precision=1e3):
        """
                          level1  level2  level3  ...
        throughput(Tb/s)     tp1     tp2     tp3  ...
        """
        throughput = defaultdict(int)
        for src in range(len(self.result.traffic_matrix)):
            for dst in range(len(self.result.traffic_matrix[src])):
                for traffic in self.result.traffic_matrix[src][dst]:
                    if not traffic.blocked:
                        throughput[traffic.security] += traffic.bandwidth / 1000    # transfer unit to Tb/s
                    else:
                        continue
        levels = sorted(list(throughput.keys()))
        throughput_to_standard_format = [levels] + [[int(throughput[key]*precision)/precision for key in levels]]
        return throughput_to_standard_format

    def analyze_link_utilization_for_each_level(self, LightPathBandwidth, precision=1e3):
        """
                            level1  level2  level3  ...
        link utilization(%)    lu1     lu2     lu3  ...
        """
        link_utilization = defaultdict(list)
        for (ori, sin, index) in self.result.MultiDiG.edges:
            link_utilization[self.result.MultiDiG[ori][sin][index]['level']].append(
                1 - self.result.MultiDiG[ori][sin][index]['bandwidth'] / LightPathBandwidth
            )
        levels = sorted(list(link_utilization.keys()))
        link_utilization_to_standard_format = [levels] + [[int(np.mean(link_utilization[level])*precision)/precision for level in levels]]
        return link_utilization_to_standard_format

    def analyze_deviation_for_each_level(self, precision=1e3):
        """
                            level1  level2  level3  ...
        level deviation(%)     ld1     ld2     ld3  ...
        """
        level_deviation = defaultdict(list)
        for src in range(len(self.result.traffic_matrix)):
            for dst in range(len(self.result.traffic_matrix[src])):
                for traffic in self.result.traffic_matrix[src][dst]:
                    delta = []
                    if not traffic.blocked:
                        for ori in traffic.lightpath:
                            delta += [(traffic.lightpath[ori]['level'] - traffic.security) ** 2]
                        level_deviation[traffic.security].append((sum(delta)/len(delta))**0.5)
                    else:
                        continue
        levels = sorted(list(level_deviation.keys()))
        level_deviation_to_standard_format = [levels] + [[int(np.mean(level_deviation[level])*precision)/precision for level in levels]]
        return level_deviation_to_standard_format

    def analyze_hop_for_each_level(self):
        """
            level1  level2  level3  ...
        hop     h1      h2      h3  ...
        """
        hop = defaultdict(list)
        for src in range(len(self.result.traffic_matrix)):
            for dst in range(len(self.result.traffic_matrix[src])):
                for traffic in self.result.traffic_matrix[src][dst]:
                    if not traffic.blocked:
                        hop[traffic.security].append(len(traffic.path)-1)
                    else:
                        continue
        levels = sorted(list(hop.keys()))
        hop_to_standard_format = [levels] + [[np.mean(hop[level]) for level in levels]]
        return hop_to_standard_format

    def analyze_success_rate_for_each_level(self):
        """
                level1  level2  level3  ...
        success     s1     s2      s3   ...
        """
        success = defaultdict(int)
        ntraffic = defaultdict(int)
        for src in range(len(self.result.traffic_matrix)):
            for dst in range(len(self.result.traffic_matrix[src])):
                for traffic in self.result.traffic_matrix[src][dst]:
                    ntraffic[traffic.security] += 1
                    if not traffic.blocked:
                        success[traffic.security] += 1
                    else:
                        continue
        levels = sorted(list(success.keys()))
        success_to_standard_format = [levels] + [[success[level]/ntraffic[level]*100 for level in levels]]
        return success_to_standard_format