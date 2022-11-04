#!/usr/bin/env python3

import pandas as pd
import numpy as np
from utils import IslType
import utils
from utils import LOGGER

class ConnectionResult():
    def __init__(self, conn_records, blocked_reqs, served_reqs, onehop_reqs):
        """
        The data frame is illustrated as below:
            "src_id" "dst_id" "bandwidth" "length" "hops" "path"
        1
        2
        3
        .......................................................
        """
        self.conns_record_df = pd.DataFrame(conn_records)
        self.blocked_reqs = blocked_reqs
        self.served_reqs = served_reqs
        self.onehop_reqs = onehop_reqs


    def get_avg_conn_len(self):
        conn_len_series = self.conns_record_df["length"]
        avg_conn_len = conn_len_series.mean()
        return avg_conn_len

    def get_avg_conn_latency(self):
        avg_conn_len = self.get_avg_conn_len()
        avg_conn_latency = utils.compute_latency_per_length(avg_conn_len)

        return avg_conn_latency

    def get_avg_conn_hops(self):
        conn_hop_series = self.conns_record_df["hops"]
        avg_conn_hops = conn_hop_series.mean()
        return avg_conn_hops

    def get_avg_conn_bw(self):
        conn_bw_series = self.conns_record_df["bandwidth"]
        avg_conn_bw = conn_bw_series.mean()
        return avg_conn_bw

    def get_blocking_ratio(self):
        overall_req_num = len(self.blocked_reqs) + len(self.served_reqs) + len(self.onehop_reqs)
        blocking_ratio = len(self.blocked_reqs)/overall_req_num
        return blocking_ratio

    def get_blocking_ratio_without_onehop(self):
        overall_req_num = len(self.blocked_reqs) + len(self.served_reqs)
        blocking_ratio = len(self.blocked_reqs)/overall_req_num
        return blocking_ratio

    def get_onehop_ratio(self):
        overall_req_num = len(self.blocked_reqs) + len(self.served_reqs) + len(self.onehop_reqs)
        onehop_ratio = len(self.onehop_reqs)/overall_req_num
        return onehop_ratio

    def metric_summary(self):
        res_dict = {}
        avg_conn_len = self.get_avg_conn_len()
        avg_conn_latency = self.get_avg_conn_latency()
        avg_conn_hops = self.get_avg_conn_hops()
        avg_conn_bw = self.get_avg_conn_bw()

        """
        blocking_ratio = self.get_blocking_ratio()
        blocking_ratio_without_onehop = self.get_blocking_ratio_without_onehop()
        onehop_ratio = self.get_onehop_ratio()

        LOGGER.info(f'Blocking ratio is : {blocking_ratio}')
        LOGGER.info(f'Blocking ratio without onehop : {blocking_ratio_without_onehop}')
        LOGGER.info(f'Onehop ratio is : {onehop_ratio}')
        """

        res_dict["avg_conn_len"] = avg_conn_len
        res_dict["avg_conn_latency"] = avg_conn_latency
        res_dict["avg_conn_hops"] = avg_conn_hops
        res_dict["avg_conn_bw"] = avg_conn_bw

        LOGGER.info(f'Average Connection Length is : {avg_conn_len}')
        LOGGER.info(f'Average Connection Latency is : {avg_conn_latency}')
        LOGGER.info(f'Average Connection Hops is : {avg_conn_hops}')
        LOGGER.info(f'Average Connection Bandwidth is : {avg_conn_bw}')

        return res_dict

class NetworkResult():
    def __init__(self, isl_cap, latest_graph, snapshotted_graph):
        self.isl_cap = isl_cap
        self.latest_graph = latest_graph
        self.snapshotted_graph = snapshotted_graph

    def get_isl_num(self):
        total_isl_num = len(self.latest_graph.edges)
        return total_isl_num

    def get_network_isl_df(self, graph):
        edge_attr_list = []
        for src, dst, data in graph.edges(data=True):
            # we dont take the RF links into account
            edge_attr_list.append(data)

        isl_df = pd.DataFrame(edge_attr_list)
        return isl_df

    #TODO, get the bw_utilization according to snapshotted_graph
    def get_bw_utilization_of_snapshoted_graph(self, snapshoted_graph):
        isl_df = self.get_network_isl_df(snapshoted_graph)
        """
        The data frame is illustrated as below:
            "length" "capacity" "isl_type"
        1
        2
        3
        .......................................................
        """
        isl_attr_df_mean = isl_df.mean()
        mean_left_capacity = isl_attr_df_mean["capacity"]
        #LOGGER.info(f'Mean Left Capacity is {mean_left_capacity}')
        bw_utilization = (self.isl_cap - mean_left_capacity) / self.isl_cap
        return bw_utilization

    def get_avg_bw_utilization(self):
        bw_util_list = []
        for item_graph in self.snapshotted_graph:
            bw_utilization = self.get_bw_utilization_of_snapshoted_graph(item_graph)
            bw_util_list.append(bw_utilization)

        bw_util_array = np.array(bw_util_list)
        avg_bw_util = bw_util_array.mean()
        return avg_bw_util

    def get_power_consumption(self):
        latest_isl_df = self.get_network_isl_df(self.latest_graph)
        intra_orbit_isl_df = latest_isl_df[latest_isl_df["isl_type"] == IslType.INTRA_ORBIT]
        intra_orbit_isl_series = intra_orbit_isl_df["length"]
        intra_orbit_isl_max_length = intra_orbit_isl_series.max()
        intra_orbit_isl_num = len(intra_orbit_isl_df)

        if not pd.isna(intra_orbit_isl_max_length):
            intra_orbit_isl_tx_power = utils.compute_isl_transmit_power(intra_orbit_isl_max_length)
        else:
            intra_orbit_isl_tx_power = 0
        #LOGGER.info(f'Transmit Power of INTRA_ORBIT ISL is {intra_orbit_isl_tx_power}')
        #LOGGER.info(f'Num of INTRA_ORBIT ISL is {intra_orbit_isl_num}')

        inter_orbit_isl_df = latest_isl_df[latest_isl_df["isl_type"] != IslType.INTER_ORBIT_IN_EIZ]
        inter_orbit_isl_series = inter_orbit_isl_df["length"]
        inter_orbit_isl_max_length = inter_orbit_isl_series.max()
        inter_orbit_isl_num = len(inter_orbit_isl_df)

        if not pd.isna(inter_orbit_isl_max_length):
            inter_orbit_isl_tx_power = utils.compute_isl_transmit_power(inter_orbit_isl_max_length)
        else:
            inter_orbit_isl_tx_power = 0

        total_isl_power = intra_orbit_isl_tx_power * intra_orbit_isl_num \
                        + inter_orbit_isl_tx_power * inter_orbit_isl_num

        return total_isl_power

    def metric_summary(self):
        res_dict = {}

        isl_num = self.get_isl_num()
        power_consumption = self.get_power_consumption()
        avg_bw_utilization = self.get_avg_bw_utilization()

        res_dict["isl_num"] = isl_num
        res_dict["pow_cons"] = power_consumption
        res_dict["bw_util"] = avg_bw_utilization

        LOGGER.info(f'Total ISL number is : {isl_num}')
        LOGGER.info(f'Total Power Consumption is : {power_consumption}')
        LOGGER.info(f'Average Bandwidth Utilization is : {avg_bw_utilization}')
        return res_dict

class CityResult():
    def __init__(self, src_conn_metrics, dst_conn_metrics):
        self.src_conn_df = pd.DataFrame.from_dict(src_conn_metrics, orient = 'index')
        self.dst_conn_df = pd.DataFrame.from_dict(dst_conn_metrics, orient = 'index')

    def set_city_pop_in_conn_df(self, city_pos):
        src_city_pop_list = []

        for city_id, _ in self.src_conn_df.iterrows():
            city_attr = city_pos[city_id]
            city_pop = city_attr["pop"]
            src_city_pop_list.append(city_pop)

        self.src_conn_df["pop"] = src_city_pop_list

        dst_city_pop_list = []
        dst_metric_top = self.dst_conn_df.head()

        for city_id, _ in self.dst_conn_df.iterrows():
            city_attr = city_pos[city_id]
            city_pop = city_attr["pop"]
            dst_city_pop_list.append(city_pop)

        self.dst_conn_df["pop"] = dst_city_pop_list

    def get_overall_blocking(self):
        """
        The data frame is illustrated as below:
              "conns_num" "conns_zerohop" "conns_nopath" "conns_nores" "pop"
        10004
        10006
        10053
        .......................................................
        """
        conn_metric_sum = self.src_conn_df.sum()

        overall_conns_num = conn_metric_sum["conns_num"]
        zerohop_conns_num = conn_metric_sum["conns_zerohop"]
        nopath_conns_num = conn_metric_sum["conns_nopath"]
        nores_conns_num = conn_metric_sum["conns_nores"]

        nopath_ratio = nopath_conns_num / overall_conns_num
        nores_ratio = nores_conns_num / overall_conns_num
        nores_ratio_without_zerohop = nores_conns_num / (overall_conns_num - zerohop_conns_num)
        zerohop_ratio = zerohop_conns_num / overall_conns_num

        return nopath_ratio, nores_ratio, nores_ratio_without_zerohop, zerohop_ratio

    def metric_summary(self):
        res_dict = {}
        nopath_ratio, nores_ratio, nores_ratio_without_zerohop, zerohop_ratio = self.get_overall_blocking()
        res_dict["nopath_ratio"] = nopath_ratio
        res_dict["nores_ratio"] = nores_ratio
        res_dict["nores_ratio_without_zerohop"] = nores_ratio_without_zerohop
        res_dict["zerohop_ratio"] = zerohop_ratio

        LOGGER.info(f'Overall Nopath Ratio is : {nopath_ratio}')
        LOGGER.info(f'Overall Nores Ratio is : {nores_ratio}')
        LOGGER.info(f'Overall Nores Ratio without Zerohop is : {nores_ratio_without_zerohop}')
        LOGGER.info(f'Overall Zerohop Ratio is : {zerohop_ratio}')

        return res_dict
