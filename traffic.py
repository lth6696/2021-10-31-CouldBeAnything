#!/usr/bin/env python3

import utils
import random
import numpy as np
import queue

from utils import ReqType

REQ_BW_RANGE = range(1, 10)

class Request():
    def __init__(self, src, dst, bw):
        self.src = src
        self.dst = dst
        self.bw = bw
        self.req_type = ReqType.UNKNOWN
        self.effective_time = None
        self.last_time = None
        self.edge_list = None

    def set_req_type(self, req_type):
        self.req_type = req_type

    def set_effective_time(self, effective_time):
        self.effective_time = effective_time

    def set_last_time(self, last_time):
        self.last_time = last_time

    def set_edge_list(self, edge_list):
        self.edge_list = edge_list

    def get_req_type(self):
        return self.req_type

    def as_leaving_req(self):
        assert self.req_type is ReqType.ARRIVAL
        self.req_type = ReqType.LEAVING
        self.effective_time += self.last_time

    def as_leaving_req_with_path(self, edge_list):
        self.as_leaving_req()
        self.set_edge_list(edge_list)

class Traffic():
    def __init__(self, city_pos_file, city_pair_file, arr_rate, lea_rate):
        self.city_positions = utils.read_city_positions(city_pos_file)
        self.city_pairs = utils.read_city_pair_file(city_pair_file)
        self.arr_rate = arr_rate
        self.lea_rate = lea_rate
        self.conn_reqs = {}

    def generate_conn_reqs(self, req_num):
        """To generate realistic connection requests, we follow the Gravity model, as below:
            1. calculate weight for corresponding city pairs according to population and distance.
            2. random choice req_num pairs per the weight generated in step 1
        """

        conn_ids = []
        city_pair_weights = []
        city_pair_probs = []

        for pair_id, city_pair in self.city_pairs.items():
            city1 = city_pair["city_1"]
            city2 = city_pair["city_2"]

            city1_pop = self.city_positions[city1]["pop"]
            city2_pop = self.city_positions[city2]["pop"]

            city_distance = city_pair["geo_dist"]


            conn_req = {
                "src_city": city1,
                "dst_city": city2,
                "city_dist": city_distance
            }
            self.conn_reqs[pair_id] = conn_req
            conn_ids.append(pair_id)

            city_pair_weight = utils.compute_gravity_weight_between_cities(city1_pop, city2_pop, city_distance)
            city_pair_weights.append(city_pair_weight)

        """Calculate the probability of each pair"""
        overall_weight = sum(city_pair_weights)
        for weight in city_pair_weights:
            city_pair_prob = weight/overall_weight
            city_pair_probs.append(city_pair_prob)

        req_ids = np.random.choice(conn_ids, req_num, p=city_pair_probs)

        reqs_in_queue = self.populate_reqs_in_queue(req_ids, req_num, self.arr_rate, self.lea_rate)

        return reqs_in_queue


    def populate_reqs_in_queue(self, req_ids, req_num, arr_lambda, lea_mu):
        reqs_in_queue = queue.PriorityQueue()
        arrival_intervals = np.random.exponential(1/arr_lambda, req_num)
        last_intervals = np.random.exponential(1/lea_mu, req_num)

        current_ref_vtime = 0
        for index in range(0, req_num):
            req_id = req_ids[index]
            can_req = self.conn_reqs[req_id]

            src = can_req["src_city"]
            dst = can_req["dst_city"]
            bw = random.choice(REQ_BW_RANGE)

            req_evt = Request(src, dst, bw)
            req_evt.set_req_type(ReqType.ARRIVAL)

            current_ref_vtime += arrival_intervals[index]
            req_evt.set_effective_time(current_ref_vtime)

            last_time = last_intervals[index]
            req_evt.set_last_time(last_time)

            reqs_in_queue.put((current_ref_vtime, req_evt))

        return reqs_in_queue
