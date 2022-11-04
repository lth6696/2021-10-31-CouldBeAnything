#!/usr/bin/env python3

import math
import networkx as nx
import random
import utils
import pdb
import pandas as pd
import random
import pdb

from utils import LOGGER
from utils import IslType
from utils import ReqType
from result import *


class Constellation():

    def __init__(self, isl_cap, sat_cov_radius):
        self.graph = nx.Graph()
        self.sat_positions = {}
        self.city_positions = {}
        self.city_sat_mappings = {}
        self.city_sat_distances = {}
        self.edge_looc_weight_df = None
        self.conn_records = []
        self.src_conn_metrics = {} #per city node
        self.dst_conn_metrics = {} #per city node

        self.isl_cap = isl_cap
        self.sat_cov_radius = sat_cov_radius
        self.orb_num = None
        self.orb_sat_num = None

        self.pendding_reqs = None
        self.onehop_reqs = []
        self.blocked_reqs = []
        self.served_reqs = []

        self.snapshotted_graph = []

    def set_pendding_reqs(self, reqs):
        self.pendding_reqs = reqs

    def add_sat_to_graph(self, sat_pos_file):
        """
        Reads satellite positions from input file
        :param sat_pos_file: input file containing satellite positions at a particular instant of time
        """
        sat_positions, orb_num, orb_sat_num = utils.read_sat_positions(sat_pos_file)
        self.orb_num = orb_num
        self.orb_sat_num = orb_sat_num
        self.sat_positions = sat_positions
        for sat_uid, sat_position in sat_positions.items():
            self.graph.add_node(sat_uid, **sat_position)
            utils.add_sat_item_to_db(sat_position)

    def add_isl_with_attr(self, src_orb_id, src_sat_id, dst_orb_id, dst_sat_id):
        src_sat = utils.get_sat_item_from_db_by_orbit_sat_id(src_orb_id, src_sat_id)
        src_sat_uid = src_sat["uid"]

        dst_sat = utils.get_sat_item_from_db_by_orbit_sat_id(dst_orb_id, dst_sat_id)
        dst_sat_uid = dst_sat["uid"]

        isl_length = utils.compute_isl_length(src_sat_uid, dst_sat_uid, self.sat_positions)

        edge_attribute = {
            "length": isl_length,
            "capacity": self.isl_cap,
            "isl_type" : IslType.UNKNOWN
        }

        self.graph.add_edge(src_sat_uid, dst_sat_uid, **edge_attribute)

    def add_isl_to_graph(self):
        """
        Add ISLs for each satellite node at four directions:
            two N-S ISLs with neighbor nodes within each orbit
            two E-W ISLs with neighbor nodes in adjacent orbits
        """
        for node_id, node_data in self.graph.nodes(data=True):
            orb_id = node_data["orb_id"]
            orb_sat_id = node_data["orb_sat_id"]

            """The North ISL: same orbit_id, orb_sat_id + 1"""
            n_isl_orb_id = orb_id
            n_isl_sat_id = (orb_sat_id + 1) % 24
            self.add_isl_with_attr(orb_id, orb_sat_id, n_isl_orb_id, n_isl_sat_id)

            """The East ISL: orbit_id + 1, same orb_sat_id"""
            e_isl_orb_id = (orb_id + 1) % 66
            e_isl_sat_id = orb_sat_id
            self.add_isl_with_attr(orb_id, orb_sat_id, e_isl_orb_id, e_isl_sat_id)

            """The South ISL: same orbit_id, orb_sat_id + 1"""
            #s_isl_orb_id = orb_id
            #s_isl_sat_id = (orb_sat_id - 1 + 66) % 66
            #self.add_isl_with_attr(orb_id, orb_sat_id, s_isl_orb_id, s_isl_sat_id)


            """The West ISL: orbit_id - 1, same orb_sat_id"""
            #w_isl_orb_id = (orb_id - 1 + 24) % 24
            #w_isl_sat_id = orb_sat_id
            #self.add_isl_with_attr(orb_id, orb_sat_id, w_isl_orb_id, w_isl_sat_id)

    def remove_isl_from_graph(self, src_sat_id, dst_sat_id):
        try:
            self.graph.remove_edge(src_sat_id, dst_sat_id)
        except Exception as e:
            LOGGER.error(f'Error occurs when removing isl between {src_sat_id} and {dst_sat_id}: {e}')

    def remove_inter_orbit_isl(self, src_orb_id, src_orb_sat_id):
        """Remove the east side ISL: orbit_id + 1, same orb_sat_id"""
        dst_orb_id = (src_orb_id + 1) % 66
        dst_orb_sat_id = src_orb_sat_id

        src_sat = utils.get_sat_item_from_db_by_orbit_sat_id(src_orb_id, src_orb_sat_id)
        src_sat_uid = src_sat["uid"]

        dst_sat = utils.get_sat_item_from_db_by_orbit_sat_id(dst_orb_id, dst_orb_sat_id)
        dst_sat_uid = dst_sat["uid"]

        self.remove_isl_from_graph(src_sat_uid, dst_sat_uid)

    def remove_intra_orbit_isl(self, src_orb_id, src_orb_sat_id):
        """Remove the north side ISL: same orbit_id, src_orb_sat_id + 1"""
        dst_orb_id = src_orb_id
        dst_orb_sat_id = (src_orb_sat_id + 1) % 24

        src_sat = utils.get_sat_item_from_db_by_orbit_sat_id(src_orb_id, src_orb_sat_id)
        src_sat_uid = src_sat["uid"]

        dst_sat = utils.get_sat_item_from_db_by_orbit_sat_id(dst_orb_id, dst_orb_sat_id)
        dst_sat_uid = dst_sat["uid"]

        self.remove_isl_from_graph(src_sat_uid, dst_sat_uid)

    def get_cities_in_range(self, sat_uid, range_radius):
        cand_city_set = set()
        sat_lat = self.sat_positions[sat_uid]["lat_deg"]
        sat_long = self.sat_positions[sat_uid]["long_deg"]

        for city_uid, city_pos in self.city_positions.items():
            city_lat = city_pos["lat_deg"]
            city_long = city_pos["long_deg"]
            sat_city_horiz_length = utils.compute_sat_city_horiz_length(sat_lat, sat_long, city_lat, city_long)
            if sat_city_horiz_length < range_radius:
                cand_city_set.add(city_uid)
        return cand_city_set

    def get_looc_weight_of_edge(self, src_sat_uid, dst_sat_uid, src_city_set, dst_city_set):
        src_city_dict = {}
        dst_city_dict = {}

        src_sat_lat = self.sat_positions[src_sat_uid]["lat_deg"]
        src_sat_long = self.sat_positions[src_sat_uid]["long_deg"]

        dst_sat_lat = self.sat_positions[dst_sat_uid]["lat_deg"]
        dst_sat_long = self.sat_positions[dst_sat_uid]["long_deg"]

        for src_city_id in src_city_set:
            src_city = self.city_positions[src_city_id]
            src_city_dict[src_city_id] = src_city
        src_city_df = pd.DataFrame.from_dict(src_city_dict, orient = 'index')

        if src_city_df.empty:
            avg_src_city_pop = 0
        else:
            avg_src_city_pop = src_city_df.mean()["pop"]

        for dst_city_id in dst_city_set:
            dst_city = self.city_positions[dst_city_id]
            dst_city_dict[dst_city_id] = dst_city
        dst_city_df = pd.DataFrame.from_dict(dst_city_dict, orient = 'index')

        if dst_city_df.empty:
            avg_dst_city_pop = 0
        else:
            avg_dst_city_pop= dst_city_df.mean()["pop"]

        src_city_weight_dict = {}
        for src_city_id, src_city_pos in src_city_dict.items():
            src_city_lat = src_city_pos["lat_deg"]
            src_city_long = src_city_pos["long_deg"]
            src_city_pop = src_city_pos["pop"]

            sat_city_length = utils.compute_sat_city_horiz_length(src_sat_lat, src_sat_long, src_city_lat, src_city_long)

            sat_city_weight = utils.compute_gravity_weight_between_cities(src_city_pop, avg_dst_city_pop, sat_city_length)
            src_sat_city_attr = {
                "length": sat_city_length,
                "weight": sat_city_weight
            }

            src_city_weight_dict[src_city_id] = src_sat_city_attr

        src_city_weight_df = pd.DataFrame.from_dict(src_city_weight_dict, orient = 'index')

        dst_city_weight_dict = {}
        for dst_city_id, dst_city_pos in dst_city_dict.items():
            dst_city_lat = dst_city_pos["lat_deg"]
            dst_city_long = dst_city_pos["long_deg"]
            dst_city_pop = dst_city_pos["pop"]

            sat_city_length = utils.compute_sat_city_horiz_length(dst_sat_lat, dst_sat_long, dst_city_lat, dst_city_long)

            sat_city_weight = utils.compute_gravity_weight_between_cities(dst_city_pop, avg_src_city_pop, sat_city_length)
            dst_sat_city_attr = {
                "length": sat_city_length,
                "weight": sat_city_weight
            }

            dst_city_weight_dict[dst_city_id] = dst_sat_city_attr

        dst_city_weight_df = pd.DataFrame.from_dict(dst_city_weight_dict, orient = 'index')

        src_city_weight_sum = 0
        dst_city_weight_sum = 0
        if not src_city_weight_df.empty:
            src_city_weight_sum = src_city_weight_df.sum()["weight"]
        if not dst_city_weight_df.empty:
            dst_city_weight_sum = dst_city_weight_df.sum()["weight"]

        return src_city_weight_sum + dst_city_weight_sum

    def pop_edge_looc_weight(self, range_radius):
        edge_weight_list = []
        for src_uid, dst_uid, edge_data in self.graph.edges(data="True"):
            src_cand_city_set = self.get_cities_in_range(src_uid, range_radius)
            dst_cand_city_set = self.get_cities_in_range(dst_uid, range_radius)

            src_city_set = src_cand_city_set - dst_cand_city_set
            dst_city_set = dst_cand_city_set - src_cand_city_set

            edge_weight = self.get_looc_weight_of_edge(src_uid, dst_uid, src_city_set, dst_city_set)

            edge_weight_tupe = (src_uid, dst_uid, edge_weight, True) # False means the link is up
            edge_weight_list.append(edge_weight_tupe)

        self.edge_looc_weight_df = pd.DataFrame(edge_weight_list, columns = ["src_uid", "dst_uid", "looc_weight", "up_flag"])

    def gravity_based_looc(self, looc_range, looc_num):
        self.pop_edge_looc_weight(looc_range)

        sorted_edge_df = self.edge_looc_weight_df.sort_values(by='looc_weight')
        #print(sorted_edge_df)

        cand_edge_df = sorted_edge_df.head(looc_num)

        for index, row in cand_edge_df.iterrows():
            src_sat_uid = row["src_uid"]
            dst_sat_uid = row["dst_uid"]
            self.remove_isl_from_graph(src_sat_uid, dst_sat_uid)


    def judge_isl_type(self, src_sat_uid, dst_sat_uid, eiz_lat):
        """
        @param:sat_uid, the uid of src sat of a isl
        @return: IslType.INTER_ORBIT_IN/OUT_EIZ
        """
        src_sat_pos = self.sat_positions[src_sat_uid]
        dst_sat_pos = self.sat_positions[dst_sat_uid]

        src_orb_id = src_sat_pos["orb_id"]
        dst_orb_id = dst_sat_pos["orb_id"]

        # different orb_id means inter_orb_isl
        if src_orb_id != dst_orb_id:
            src_sat_id = src_sat_pos["orb_sat_id"]
            dst_sat_id = dst_sat_pos["orb_sat_id"]
            assert src_sat_id == dst_sat_id

            src_sat_lat = src_sat_pos["lat_deg"]
            dst_sat_lat = dst_sat_pos["lat_deg"]

            if abs(src_sat_lat) < eiz_lat or abs(dst_sat_lat) < eiz_lat:
                return IslType.INTER_ORBIT_IN_EIZ
            else:
                return IslType.INTER_ORBIT_OUT_EIZ
        # same orb_id means intra_orb_isl
        else:
            return IslType.INTRA_ORBIT

    def mark_isl_type_per_eiz(self, eiz_lat_deg):
        for src_sat_uid, dst_sat_uid, edge_data in self.graph.edges(data=True):
            isl_type = self.judge_isl_type(src_sat_uid, dst_sat_uid, eiz_lat_deg)
            edge_data["isl_type"] = isl_type

    """Switching ON/OFF the inter-orbit ISLs according to EIZ"""
    def eiz_based_looc(self, eiz_lat_deg):
        self.mark_isl_type_per_eiz(eiz_lat_deg)

        cand_isl_list = []

        for src_sat_uid, dst_sat_uid, edge_data in self.graph.edges(data=True):
            if edge_data["isl_type"] == IslType.INTER_ORBIT_IN_EIZ:
                # in use deletion might occurs, lets see
                cand_isl_list.append((src_sat_uid, dst_sat_uid))

        for cand_isl in cand_isl_list:
            self.remove_isl_from_graph(cand_isl[0], cand_isl[1])

        LOGGER.info(f'Number of closed ISLs in EIZ is {len(cand_isl_list)}')

    def calc_adaptive_orb_shift(self, orb_sat_num, orb_sat_off_num):
        """
        __________________________________________________________
        OFF ON ON ON OFF ON ON ON OFF ON ON ON OFF ON ON ON
        |shift space|

        """
        if orb_sat_off_num == 0:
            return 0
        float_shift_space = orb_sat_num / orb_sat_off_num
        float_half_shift = float_shift_space / 2

        floored_half_shift = (int) (float_half_shift)
        left_float_shift = float_half_shift - floored_half_shift

        # we determie if we floor or ceil the float_half_shift as the final int_shift
        rand_chance = random.random()

        # we floor the float_shift if the left_float_shift is less than the rand_chance
        if left_float_shift < rand_chance:
            return floored_half_shift
        else:
            return floored_half_shift+1

    def spherically_symmetric_looc(self, sat_off_ratio, orb_off_ratio):
        cand_src_sats = {}
        last_sat_ids = []

        sat_off_num = (int) (self.orb_sat_num * sat_off_ratio)
        sat_off_mask = utils.near_even_select(self.orb_sat_num, sat_off_num)
        print(sat_off_mask)

        orb_off_num = (int) (self.orb_num * orb_off_ratio)
        orb_off_mask = utils.near_even_select(self.orb_num, orb_off_num)
        print(orb_off_mask)

        orb_shift = self.calc_adaptive_orb_shift(self.orb_sat_num, sat_off_num)
        print(orb_shift)

        for cand_sat_id in range(0, self.orb_sat_num):
            if not sat_off_mask[cand_sat_id]:
                last_sat_ids.append(cand_sat_id)

        for cand_orb_id in range(0, self.orb_num):
            if not orb_off_mask[cand_orb_id]:
                cand_sat_ids = []
                for cand_sat_id in last_sat_ids:
                    shifted_sat_id = (cand_sat_id + orb_shift) % self.orb_sat_num
                    cand_sat_ids.append(shifted_sat_id)
                cand_src_sats[cand_orb_id] = cand_sat_ids
                last_sat_ids = cand_sat_ids

        print(cand_src_sats)

        num_of_closed_isl = 0
        for orb_id, orb_sat_id_list in cand_src_sats.items():
            for orb_sat_id in orb_sat_id_list:
                self.remove_intra_orbit_isl(orb_id, orb_sat_id)
                num_of_closed_isl += 1

        LOGGER.info(f'Number of closed ISLs of SSLOOC is {num_of_closed_isl}')

    def add_city_to_constellation(self, city_pos_file):
        city_positions = utils.read_city_positions(city_pos_file)
        self.city_positions = city_positions

        for city_uid, city_position in city_positions.items():
            #self.graph.add_node(city_uid, **city_position)
            #utils.add_city_item_to_db(city_position)

            src_conn_metrics = {
                "conns_num": 0,
                "conns_zerohop": 0,
                "conns_nopath": 0,
                "conns_nores": 0
            }

            dst_conn_metrics = {
                "conns_num": 0,
                "conns_zerohop": 0,
                "conns_nopath": 0,
                "conns_nores": 0
            }

            self.src_conn_metrics[city_uid] = src_conn_metrics
            self.dst_conn_metrics[city_uid] = dst_conn_metrics

    #def add_city_sat_mappings(self, coverage_file):
    #    """
    #    Adds city-sat mappings
    #    :param graph: The graph under consideration with nodes being the satellites and/or cities
    #    :param city: The city for which coverage needs to be added
    #    :param city_coverage: Collection of city-satellite mappings
    #    """
    #    city_coverages = utils.read_city_coverage(coverage_file)
    """
        for _, coverage in city_coverages.items():
            city_id = coverage["city"]
            sat_id = coverage["sat"]
            mapping_attribute = {
                "length": coverage["dist"]
            }

            if city_id in self.city_sat_mappings:
                sat_mappings = self.city_sat_mappings[city_id]
                sat_mappings[sat_id] = mapping_attribute
            else:
                sat_mappings = {}
                sat_mappings[sat_id] = mapping_attribute
                self.city_sat_mappings[city_id] = sat_mappings
    """

    def add_sat_city_length_to_dict(self, sat_uid, city_id, length, target_dict):
        mapping_attribute = {
            "length": length
        }

        if city_id in target_dict:
            sat_mappings = target_dict[city_id]
            sat_mappings[sat_uid] = mapping_attribute
        else:
            sat_mappings = {}
            sat_mappings[sat_uid] = mapping_attribute
            target_dict[city_id] = sat_mappings


    def add_city_sat_mappings(self):
        """
        Add city-sat associations, according to the latitude/longtitude data of sat/city
        """
        for city_id, city_pos in self.city_positions.items():
            city_lat = city_pos["lat_deg"]
            city_long = city_pos["long_deg"]

            for sat_uid, sat_pos in self.sat_positions.items():
                sat_lat = sat_pos["lat_deg"]
                sat_long = sat_pos["long_deg"]

                horiz_length = utils.compute_sat_city_horiz_length(sat_lat, sat_long, city_lat, city_long)

                self.add_sat_city_length_to_dict(sat_uid, city_id, horiz_length, self.city_sat_distances)

                if horiz_length < self.sat_cov_radius:
                    self.add_sat_city_length_to_dict(sat_uid, city_id, horiz_length, self.city_sat_mappings)

    def map_city_to_sat(self, src_city_id, dst_city_id):
        if src_city_id in self.city_sat_mappings and dst_city_id in self.city_sat_mappings:
            src_sat_mappings = self.city_sat_mappings[src_city_id]
            dst_sat_mappings = self.city_sat_mappings[dst_city_id]

            src_sat_id = random.choice(list(src_sat_mappings.keys()))
            dst_sat_id = random.choice(list(dst_sat_mappings.keys()))

            return src_sat_id, dst_sat_id
        else:
            return None, None

    def release_connection(self, req):

        conn_bw = req.bw
        edges_in_path = req.edge_list

        if not edges_in_path:
            LOGGER.error('No conn path found, this is considered as a programming error')
            return None

        for edge in edges_in_path:
            edge["capacity"] += conn_bw

    def allocate_connection(self, req):
        """
        To provision a connection, we:
            1. calculate the shortest path
            2. allocate the required bandwidth from each hop along the path
            3. store the connection details to the db and mem
        """
        src_city_uid = req.src
        dst_city_uid = req.dst
        bandwidth = req.bw

        src_conn_metrics = self.src_conn_metrics[src_city_uid]
        dst_conn_metrics = self.dst_conn_metrics[dst_city_uid]

        src_conn_metrics["conns_num"] += 1
        dst_conn_metrics["conns_num"] += 1

        #LOGGER.info(f'Handling connection request from: {src_city_uid} to {dst_city_uid}')

        src_sat_uid, dst_sat_uid = self.map_city_to_sat(src_city_uid, dst_city_uid)
        if src_sat_uid == None or dst_sat_uid == None:
            LOGGER.warn('No up/down links for one more connection request')
            return None

        # list
        try:
            shortest_path = nx.shortest_path(self.graph, src_sat_uid, dst_sat_uid, "length")
        except Exception as e:
            src_conn_metrics["conns_nopath"] += 1
            dst_conn_metrics["conns_nopath"] += 1
            #LOGGER.info(f'No shortest path to {dst_sat_uid}')
            return None

        if not shortest_path:
            LOGGER.error('No shortest path found, this is considered as a programming error')
            return None
        elif len(shortest_path) == 1:
            #pdb.set_trace()
            src_conn_metrics["conns_zerohop"] += 1
            dst_conn_metrics["conns_zerohop"] += 1
            self.onehop_reqs.append(req)
            #LOGGER.info(f'Zero hop between {src_sat_uid} and {dst_sat_uid}, no need for ISL provisioning')
            return None


        edges_in_path = []
        pre_node = None
        for node in shortest_path:
            if not pre_node:
                pre_node = node
                continue
            else:
                traverse_edge = self.graph[pre_node][node]
                pre_node= node
                edges_in_path.append(traverse_edge)

        """Allocate bandwidth only if all the traverse edges have sufficient capacity"""

        #pdb.set_trace()
        for edge in edges_in_path:
            edge_capacity = edge["capacity"]
            if bandwidth > edge_capacity:
                #LOGGER.info(f'Insufficient bandwidth: {edge_capacity}')
                src_conn_metrics["conns_nores"] += 1
                dst_conn_metrics["conns_nores"] += 1
                self.blocked_reqs.append(req)
                return None
            else:
                continue

        """Till here, the bandwdith requirment can be met, start to allocate bandwidth"""
        path_length = 0
        path_hops = 0
        for edge in edges_in_path:
            edge["capacity"] -= bandwidth
            path_length += edge["length"]
            path_hops += 1

        conn_record = {
            "src_id": src_sat_uid,
            "dst_id": dst_sat_uid,
            "bandwidth": bandwidth,
            "length": path_length,
            "hops": path_hops,
            "path": shortest_path
        }

        self.conn_records.append(conn_record)
        self.served_reqs.append(req)
        #utils.add_conn_rec_to_db(conn_record)
        return edges_in_path

    def collect_sim_results(self):
        overall_res = {}
        net_result = NetworkResult(self.isl_cap, self.graph, self.snapshotted_graph)
        net_res = net_result.metric_summary()
        overall_res.update(net_res)

        conn_result = ConnectionResult(self.conn_records, self.blocked_reqs, self.served_reqs, self.onehop_reqs)
        conn_res = conn_result.metric_summary()
        overall_res.update(conn_res)

        city_result = CityResult(self.src_conn_metrics, self.dst_conn_metrics)
        city_result.set_city_pop_in_conn_df(self.city_positions)
        city_res = city_result.metric_summary()
        overall_res.update(city_res)

        return overall_res

    def get_bw_sample_points(self):
        """ We sample the middle 60% of all the reqs"""
        sample_points = []
        req_num = self.pendding_reqs.qsize()
        for i in range (3, 8):
            sample_points.append(req_num * i / 100)

        return sample_points


    def sim_run(self, arr_rate):
        processed_req_num = 0
        load_stacked_res = {}
        load_stacked_res["arr_rate"] = arr_rate

        bw_sample_points = self.get_bw_sample_points()

        while True:
            if self.pendding_reqs.empty():
                break

            req_item = self.pendding_reqs.get()
            req = req_item[1]
            req_type = req.get_req_type()

            if req_type is ReqType.ARRIVAL:
                edges_in_path = self.allocate_connection(req)
                processed_req_num += 1

                if processed_req_num in bw_sample_points:
                    self.snapshotted_graph.append(self.graph.copy())
                    LOGGER.info(f'Snapshoting the graph for the {processed_req_num}th requests')

                if edges_in_path:
                    req.as_leaving_req_with_path(edges_in_path)
                    self.pendding_reqs.put((req.effective_time, req))

            elif req_type is ReqType.LEAVING:
                self.release_connection(req)
            else:
                LOGGER.error(f'UNKNOWN request type, this should be considered as programming error')

        snapshoted_res = self.collect_sim_results()
        load_stacked_res.update(snapshoted_res)
        return load_stacked_res
