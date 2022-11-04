#!/usr/bin/env python3

import os
from constellation import Constellation
from traffic import Traffic
import utils
from utils import LOGGER
import pandas as pd
import numpy as np
from pandas import DataFrame as df

SAT_LOCATION_FILE = './data/starlink_positions.txt'
CITY_LOCATION_FILE = './data/cities.txt'
CITY_COVERAGE_FILE = './data/city_coverage_0.txt'
CITY_PAIR_FILE = './data/city_pairs.txt'

LEO_ALTITUDE = 530
LEO_ELE_ANGLE = 25

ISL_CAP = 10000

EIZ_LATITUDE = 2
LEA_RATE = 0.01

LOOC_RANGE = 2000

NUM_OF_REQUESTS = 500000

def init_sat_constellation():
    sat_cov_radius = utils.calc_coverage_radius(LEO_ALTITUDE, LEO_ELE_ANGLE)
    starlink = Constellation(ISL_CAP, sat_cov_radius)

    starlink.add_sat_to_graph(SAT_LOCATION_FILE)
    starlink.add_isl_to_graph()

    starlink.add_city_to_constellation(CITY_LOCATION_FILE)
    starlink.add_city_sat_mappings()

    return starlink

def dump_res_dict_to_csv(res_dict, csv_path, csv_name):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    print(res_dict)
    res_df = df(data=res_dict, index=[0])
    csv_file_path = os.path.join(csv_path, csv_name)
    if not os.path.isfile(csv_file_path):
        res_df.to_csv(csv_file_path, header=True, index=False)
    else: # else it exists so append without writing the header
        res_df.to_csv(csv_file_path, mode='a', header=False, index=False)

def gravity_based_looc_simulation(looc_range, looc_num, arr_rate):
    LOGGER.info('------------------------------------------------------------------')
    LOGGER.info(f'Simulating with looc_range: {looc_range} and looc_num: {looc_num}')
    utils.clear_all_dbs()

    starlink = init_sat_constellation()
    traffic = Traffic(CITY_LOCATION_FILE, CITY_PAIR_FILE, arr_rate, LEA_RATE)
    reqs = traffic.generate_conn_reqs(NUM_OF_REQUESTS)

    starlink.gravity_based_looc(looc_range, looc_num)
    starlink.set_pendding_reqs(reqs)
    sim_res = starlink.sim_run(arr_rate)

    csv_path = "gravity_based_looc"
    csv_name = f'{looc_range}_{looc_num}.csv'
    dump_res_dict_to_csv(sim_res, csv_path, csv_name)

def eiz_based_looc_simulation(eiz_lat_deg, arr_rate):
    LOGGER.info('------------------------------------------------------------------')
    LOGGER.info(f'Simulating with eiz_lat_deg: {eiz_lat_deg}')
    utils.clear_all_dbs()

    starlink = init_sat_constellation()
    traffic = Traffic(CITY_LOCATION_FILE, CITY_PAIR_FILE, arr_rate, LEA_RATE)
    reqs = traffic.generate_conn_reqs(NUM_OF_REQUESTS)

    starlink.eiz_based_looc(eiz_lat_deg)
    starlink.set_pendding_reqs(reqs)
    sim_res = starlink.sim_run(arr_rate)

    csv_path = "eiz_based_looc"
    csv_name = f'{eiz_lat_deg}.csv'
    dump_res_dict_to_csv(sim_res, csv_path, csv_name)

def spherically_symmetric_looc_simulation(sat_off_ratio, orb_off_ratio, arr_rate):
    LOGGER.info('------------------------------------------------------------------')
    LOGGER.info(f'Simulating with orb_off_ratio: {orb_off_ratio}, sat_off_ratio: {sat_off_ratio}')
    utils.clear_all_dbs()

    starlink = init_sat_constellation()
    traffic = Traffic(CITY_LOCATION_FILE, CITY_PAIR_FILE, arr_rate, LEA_RATE)
    reqs = traffic.generate_conn_reqs(NUM_OF_REQUESTS)
    starlink.mark_isl_type_per_eiz(0)

    starlink.spherically_symmetric_looc(sat_off_ratio, orb_off_ratio)
    starlink.set_pendding_reqs(reqs)
    sim_res = starlink.sim_run(arr_rate)

    csv_path = "spherically_symmetric_looc"
    csv_name = f'{sat_off_ratio}_{orb_off_ratio}.csv'
    dump_res_dict_to_csv(sim_res, csv_path, csv_name)

#if __name__ == "__main__":
utils.init_logger()
"""
for looc_range in range(500, 3000, 100):
    for looc_num in range(0, 1000, 100):
        gravity_based_simulation(looc_range, looc_num)
"""
"""
for orb_gap in range(1, 10):
    for sat_gap in range(1, 10):
        for orb_shift in range(0, sat_gap):
            spherically_symmetric_looc_simulation(orb_gap, sat_gap, orb_shift)
"""
for sat_off_ratio in np.linspace(0.1, 0.9, 9).round(decimals=1):
    for orb_off_ratio in np.linspace(0.1, 0.9, 9).round(decimals=1):
        for arr_rate in range(50, 100, 10):
            spherically_symmetric_looc_simulation(sat_off_ratio, orb_off_ratio, arr_rate)

#for eiz_sat_deg in range(0, 50, 10):
#    for arr_rate in range(80, 150, 10):
#        eiz_based_looc_simulation(eiz_sat_deg, arr_rate)
