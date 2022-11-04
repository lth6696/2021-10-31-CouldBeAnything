#!/usr/bin/env python3

import math
import pymongo
import numpy as np
import logging

from enum import Enum

class IslType(Enum):
    UNKNOWN = 0
    INTRA_ORBIT = 1
    INTER_ORBIT_IN_EIZ = 2
    INTER_ORBIT_OUT_EIZ = 3

class ReqType(Enum):
    UNKNOWN = 0
    ARRIVAL = 1
    LEAVING = 2

EARTH_RADIUS = 6371  # Kms

# ISL Channel Modeling
SNR = 1000
LAMBDA = 1.064 * math.pow(10, -6)
BW = 10 * math.pow(10, 9)
N0 = math.pow(10, -17)
PI = math.pi
DR = 0.125
WAIST = 5 * math.pow(10, -3)

DB_CLIENT = pymongo.MongoClient(host='localhost', port=27017)
SAT_DB = DB_CLIENT["satnet"]

SAT_LOC = SAT_DB["satloc"]
CITY_LOC = SAT_DB["cityloc"]
CONN_REC = SAT_DB["connrec"]
LOGGER = logging.getLogger('salacom')

def init_logger():
    LOGGER.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('./salacom.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

def add_sat_item_to_db(sat_location):
    SAT_LOC.insert_one(sat_location)

def get_sat_item_from_db_by_orbit_sat_id(orb_id, orb_sat_id):
    search_index = {"orb_id": orb_id,
                    "orb_sat_id": orb_sat_id}
    return SAT_LOC.find_one(search_index)

def clear_sat_db():
    SAT_LOC.delete_many({})

def add_city_item_to_db(city_location):
    CITY_LOC.insert_one(city_location)

def clear_city_db():
    CITY_LOC.delete_many({})

def add_conn_rec_to_db(conn_rec):
    CONN_REC.insert_one(conn_rec)

def clear_conn_rec_db():
    CONN_REC.delete_many({})

def clear_all_dbs():
    clear_sat_db()
    clear_city_db()
    clear_conn_rec_db()

def compute_isl_length(sat1, sat2, sat_positions):
    """
    Computes ISL length between pair of satellites. This function can also be used to compute
    city-satellite up/down-link length.
    :param sat1: Satellite 1 with position information (latitude, longitude, altitude)
    :param sat2: Satellite 2 with position information (latitude, longitude, altitude)
    :param sat_positions: Collection of satellites along with their current position data
    :return: ISl length in km
    """
    x1 = (EARTH_RADIUS + sat_positions[sat1]["alt_km"]) * math.cos(sat_positions[sat1]["lat_rad"]) * math.sin(
        sat_positions[sat1]["long_rad"])
    y1 = (EARTH_RADIUS + sat_positions[sat1]["alt_km"]) * math.sin(sat_positions[sat1]["lat_rad"])
    z1 = (EARTH_RADIUS + sat_positions[sat1]["alt_km"]) * math.cos(sat_positions[sat1]["lat_rad"]) * math.cos(
        sat_positions[sat1]["long_rad"])
    x2 = (EARTH_RADIUS + sat_positions[sat2]["alt_km"]) * math.cos(sat_positions[sat2]["lat_rad"]) * math.sin(
        sat_positions[sat2]["long_rad"])
    y2 = (EARTH_RADIUS + sat_positions[sat2]["alt_km"]) * math.sin(sat_positions[sat2]["lat_rad"])
    z2 = (EARTH_RADIUS + sat_positions[sat2]["alt_km"]) * math.cos(sat_positions[sat2]["lat_rad"]) * math.cos(
        sat_positions[sat2]["long_rad"])
    dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2) + math.pow((z2 - z1), 2))
    return dist

def compute_sat_city_horiz_length(sat_lat, sat_long, city_lat, city_long):
    """
    Computes horizontal length between pair of satellites.
    :return: length in km
    """
    sat_lat_rad = math.radians(sat_lat)
    sat_long_rad = math.radians(sat_long)

    city_lat_rad = math.radians(city_lat)
    city_long_rad = math.radians(city_long)

    x1 = EARTH_RADIUS * math.cos(sat_lat_rad) * math.sin(sat_long_rad)
    y1 = EARTH_RADIUS * math.sin(sat_lat_rad)
    z1 = EARTH_RADIUS * math.cos(sat_lat_rad) * math.cos(sat_long_rad)
    x2 = EARTH_RADIUS * math.cos(city_lat_rad) * math.sin(city_long_rad)
    y2 = EARTH_RADIUS * math.sin(city_lat_rad)
    z2 = EARTH_RADIUS * math.cos(city_lat_rad) * math.cos(city_long_rad)
    dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2) + math.pow((z2 - z1), 2))
    return dist

def compute_gravity_weight_between_cities(city1_pop, city2_pop, city_distance):
    gravity_weight = city1_pop * city2_pop / (city_distance ** 2)
    return gravity_weight

def compute_isl_transmit_power(isl_length):
    isl_length_in_meter = isl_length * 1000
    transmit_power = SNR * BW * N0 / (1 - np.exp(-2 * (DR ** 2) * (PI ** 2) * (WAIST ** 2) / ((PI ** 2) * (WAIST ** 4) + (isl_length_in_meter ** 2) * (LAMBDA ** 2))))
    return transmit_power

def compute_latency_per_length(conn_length):
    latency = conn_length * 1000 / (3 * math.pow(10, 8))
    return latency

def read_sat_positions(sat_pos_file):
    """
    reads city coordinates and population
    :param city_pos_file: file containing city coordinates and population
    :return: collection of cities with coordinates and populations
    """

    sat_positions = {}
    max_orb_id = 0
    max_orb_sat_id = 0
    lines = [line.rstrip('\n') for line in open(sat_pos_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        sat_position = {
            "uid": int(val[0]),
            "orb_id": int(val[1]),
            "orb_sat_id": int(val[2]),
            "lat_deg": float(val[4]),
            "lat_rad": math.radians(float(val[4])),
            "long_deg": float(val[3]),
            "long_rad": math.radians(float(val[3])),
            "alt_km": float(val[5])
        }
        sat_positions[int(val[0])] = sat_position
        orb_id = int(val[1])
        if orb_id > max_orb_id:
            max_orb_id = orb_id

        orb_sat_id = int(val[2])
        if orb_sat_id > max_orb_sat_id:
            max_orb_sat_id = orb_sat_id
    return sat_positions, max_orb_id + 1, max_orb_sat_id + 1

def read_city_positions(city_pos_file):
    """
    reads city coordinates and population
    :param city_pos_file: file containing city coordinates and population
    :return: collection of cities with coordinates and populations
    """
    city_positions = {}
    lines = [line.rstrip('\n') for line in open(city_pos_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_positions[int(val[0])] = {
            "uid": int(val[0]),
            "lat_deg": float(val[2]),
            "long_deg": float(val[3]),
            "pop": float(val[4])
        }
    return city_positions


def read_city_coverage(coverage_file):
    """
    Reads city covrage in terms of which satellites a city can communicate with
    :param coverage_file: This file holds the city-satellite mapping along with the up/down-link distance in km
    :return: Collection of city-satellite mappings
    """
    city_coverage = {}
    lines = [line.rstrip('\n') for line in open(coverage_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_coverage[i] = {
            "city": int(val[0]),
            "sat": int(val[1]),
            "dist": float(val[2])
        }
    return city_coverage


def read_city_pair_file(city_pair_file):
    """
    Reads the city pairs to be considered while computing the performance metric.
    :param city_pair_file:  This file contains city pairs and the geodesic distance in km between them
    :return: Collection of city-city geodesic distances
    """
    city_pairs = {}
    lines = [line.rstrip('\n') for line in open(city_pair_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_pairs[i] = {
            "city_1": int(val[0]),
            "city_2": int(val[1]),
            "geo_dist": float(val[2])
        }
    return city_pairs


def get_neighbor_satellite(sat1_orb, sat1_rel_id, sat2_orb, sat2_rel_id, sat_positions, num_orbits, num_sats_per_orbit):
    """
    Get the absolute id of the neighbor satellite from relative ids
    :param sat1_orb: orbit id of satellite 1
    :param sat1_rel_id: relative id within orbit for satellite 1
    :param sat2_orb: orbit id of satellite 2
    :param sat2_rel_id: relative id within orbit for satellite 2
    :param sat_positions: Collection of satellites with position data
    :param num_orbits: Number of orbits in the constellation
    :param num_sats_per_orbit: Number of satellites per orbit
    :return: absolute id of the neighbor satellite
    """
    neighbor_abs_orb = (sat1_orb + sat2_orb) % num_orbits
    neighbor_abs_pos = (sat1_rel_id + sat2_rel_id) % num_sats_per_orbit
    sel_sat_id = -1
    for sat_id in sat_positions:
        if sat_positions[sat_id]["orb_id"] == neighbor_abs_orb \
                and sat_positions[sat_id]["orb_sat_id"] == neighbor_abs_pos:
            sel_sat_id = sat_id
            break
    return sel_sat_id


def check_edge_availability(graph, node1, node2):
    """
    Checks if an edge between 2 satellites is possible considering each satellite can have at most 4 ISLs
    :param graph: The graph under consideration with nodes being the satellites and/or cities
    :param node1: Denotes satellite 1
    :param node2: Denotes satellite 2
    :return: Decision whether an ISL is possible given the current degrees of the 2 nodes
    """
    is_possible = True
    deg1 = graph.degree(node1)
    deg2 = graph.degree(node2)
    if deg1 > 3 or deg2 > 3:
        is_possible = False
    return is_possible


def remove_coverage_for_city(graph, city, city_coverage):
    """
    Removes city-satellite up/down-links from the graph
    :param grph: The graph under consideration with nodes being the satellites and/or cities
    :param city: The city for which coverage needs to be removed
    :param city_coverage: Collection of city-satellite mappings
    :return: Updated graph
    """
    for i in range(len(city_coverage)):
        if city_coverage[i]["city"] == city:
            graph.remove_edge(city_coverage[i]["city"], city_coverage[i]["sat"])
    return graph

def calc_coverage_radius(altitude, ele_angle):
    """
    calculate the radius of the coverage area of one satellite
    :param altitude: the altitude of a satellite
    :param ele_angle: min elevation angle of a satellite terminal
    :return: radius of coverage area
    """
    X = EARTH_RADIUS * math.cos(math.radians(ele_angle)) / (EARTH_RADIUS + altitude)
    cov_rad = EARTH_RADIUS * (PI / 2 - math.radians(ele_angle) - math.asin(X))
    return cov_rad

def calc_mean_value_of_list(sample_list):
    """
    calculate the mean value of a numeric list
    :param sample_list: the list of sample values
    :return: mean value of the list
    """
    if not sample_list:
        return None
    else:
        return np.mean(sample_list)

def calc_median_value_of_list(sample_list):
    """
    calculate the median value of a numeric list
    :param sample_list: the list of sample values
    :return: median value of the list
    """
    if not sample_list:
        return None
    else:
        return np.median(sample_list)

def near_even_select(sample_num, select_num):
    if select_num == 0:
        selection_mask = np.ones(sample_num, dtype=int)
        return selection_mask
    if select_num > sample_num/2:
        selection_mask = np.zeros(sample_num, dtype=int)
        q, r = divmod(sample_num, sample_num-select_num)
        indices = [q*i + min(i, r) for i in range(sample_num-select_num)]
        selection_mask[indices] = True
    else:
        selection_mask = np.ones(sample_num, dtype=int)
        q, r = divmod(sample_num, select_num)
        indices = [q*i + min(i, r) for i in range(select_num)]
        selection_mask[indices] = False
    return selection_mask
