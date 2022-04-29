import networkx as nx
from collections import defaultdict


class Result(object):
    def __init__(self):
        self.num_traffic_success_mapping = 0
        self.num_traffic_fail_mapping = 0
        self.num_traffic = 0
        self.traffic_mapping_success_rate = 0
        self.set_block_traffic = {}
        self.set_success_traffic = {}

    def set_attrs(self, nsuc: int, nfail: int, nt: int, setblock: dict, setsuccess: dict):
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

        self._verify_quantity_correct()

    def _get_success_mapping_rate(self):
        if self.num_traffic == 0:
            raise Exception('The number of traffic can not be zero.')
        return self.num_traffic_success_mapping / self.num_traffic * 100

    def _verify_quantity_correct(self):
        if self.num_traffic != self.num_traffic_success_mapping + self.num_traffic_fail_mapping:
            raise Exception('Quantity relationship is wrong.')


class ResultAnalysisImpl(object):
    def __init__(self):
        pass

    def analysis_traffic_block_rate_under_different_situation(self, result: Result):
        analysis_result = {}    # {level1: {'0x01': int, '0x02': int, ...}, ...}
        for key in result.set_block_traffic.keys():
            analysis_result[key] = defaultdict(int)
            for traffic in result.set_block_traffic[key]:
                analysis_result[key][traffic['block_reason']] += 1
        """
                situation1  situation2  ...
        level1        0.0%        0.0%  ...
        level2        0.0%        0.0%  ...
           ...         ...         ...  ...
        """
        situations = set()
        for key in analysis_result:
            situations = situations | set(analysis_result[key].keys())
        situations = sorted(situations)
        analysis_result_to_matrix = [[] for _ in range(len(analysis_result.keys()))]
        for level in analysis_result.keys():
            for situation in situations:
                if situation in analysis_result[level].keys():
                    analysis_result_to_matrix[level-1].append(
                        analysis_result[level][situation] / result.num_traffic * 100
                    )
                else:
                    analysis_result_to_matrix[level - 1].append(0.0)
        return analysis_result_to_matrix
