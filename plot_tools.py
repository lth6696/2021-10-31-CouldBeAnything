#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SAT_OFF_RATIO_RANGE = np.linspace(0.1, 0.8, 8).round(decimals=1)
ORB_OFF_RATIO_RANGE = np.linspace(0.1, 0.8, 8).round(decimals=1)
EIZ_LAT_RANGE = range(0, 50, 10)

ARR_RATE_RANGE = range(80, 150, 10)

eiz_based_looc_result = {}
spherically_symmetric_looc_result = {}

def load_results_df_from_csv(csv_name):
    if os.path.exists(csv_name):
        res_df = pd.read_csv(csv_name, index_col=0)
        return res_df
    else:
        return None


def load_eiz_based_looc_result_from_csv(eiz_lat_range):

    csv_path = "eiz_based_looc"

    for eiz_lat in eiz_lat_range:
        csv_name = f'{eiz_lat}.csv'
        csv_file_path = os.path.join(csv_path, csv_name)
        res_df = load_results_df_from_csv(csv_file_path)

        eiz_based_looc_result[eiz_lat] = res_df

def load_spherically_symmetric_looc_result_from_csv(sat_off_ratio_range, orb_off_ratio_range):

    csv_path = "spherically_symmetric_looc"

    for sat_off_ratio in sat_off_ratio_range:
        if sat_off_ratio not in spherically_symmetric_looc_result:
            spherically_symmetric_looc_result[sat_off_ratio] = {}
        result_per_sat_off_ratio = spherically_symmetric_looc_result[sat_off_ratio]
        for orb_off_ratio in orb_off_ratio_range:
            csv_name = f'{sat_off_ratio}_{orb_off_ratio}.csv'
            csv_file_path = os.path.join(csv_path, csv_name)
            res_df = load_results_df_from_csv(csv_file_path)
            result_per_sat_off_ratio[orb_off_ratio] = res_df

def plot_metric_against_sat_off_ratio_and_orb_off_ratio(sat_off_ratio_range, orb_off_ratio_range, arr_rate, metric_label):
    """
    This func plot a multiple line graph with orb_shift as x-axis
    We keep the sat_gap as const, and Each line corresponds to a orb_gap
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sat_off_ratio_tickers =  np.array(sat_off_ratio_range)
    for sat_off_ratio_y in sat_off_ratio_tickers:
        orb_off_ratio_tickers = np.array(orb_off_ratio_range)
        bar_value_array = []
        for orb_off_ratio_x in orb_off_ratio_tickers:
            bar_value_array.append(spherically_symmetric_looc_result[sat_off_ratio_y][orb_off_ratio_x].loc[arr_rate, metric_label])
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax.bar(orb_off_ratio_tickers, bar_value_array, zs=sat_off_ratio_y, zdir='y', width=0.05)


    ax.set_xlabel('orb_off_ratio')
    ax.set_ylabel('sat_off_ratio')
    ax.set_zlabel(metric_label)

    ax.set_yticks(sat_off_ratio_tickers)

    plt.legend()
    plt.show()


def plot_metric_against_eiz_lat(eiz_lat_range, arr_rate_range, metric_label):
    """
    This func plot a bar graph with eiz_lat as x-axis
    """
    fig, ax = plt.subplots()
    ax.set_title(f"{metric_label}_over_eiz_lat")
    ax.set_ylabel(f"{metric_label}")
    ax.set_xlabel("eiz_lat")

    bar_width = 0.85
    pos_shift = -len(arr_rate_range)/2 * bar_width

    X_eiz_lat = np.array(eiz_lat_range)

    for arr_rate in arr_rate_range:
        Y_value_array = []
        for eiz_lat in X_eiz_lat:
            Y_value = eiz_based_looc_result[eiz_lat].loc[arr_rate, metric_label]
            Y_value_array.append(Y_value)
        ax.bar(X_eiz_lat+pos_shift, Y_value_array, bar_width, label=arr_rate)
        pos_shift += bar_width

    plt.legend()
    plt.show()

def plot_metric_against_arr_rate(eiz_lat_range, arr_rate_range, metric_label):
    """
    This func plot a bar graph with arr_rate as x-axis
    """
    fig, ax = plt.subplots()
    ax.set_title(f"{metric_label}_over_arr_rate")
    ax.set_ylabel(f"{metric_label}")
    ax.set_xlabel("arr_rate")

    bar_width = 0.85
    pos_shift = -len(eiz_lat_range)/2 * bar_width

    X_arr_rate = np.array(arr_rate_range)

    for eiz_lat in eiz_lat_range:
        Y_value_array = []
        for arr_rate in X_arr_rate:
            Y_value = eiz_based_looc_result[eiz_lat].loc[arr_rate, metric_label]
            Y_value_array.append(Y_value)
        ax.bar(X_arr_rate+pos_shift, Y_value_array, bar_width, label=eiz_lat)
        pos_shift += bar_width

    plt.legend()
    plt.show()

load_spherically_symmetric_looc_result_from_csv(SAT_OFF_RATIO_RANGE, ORB_OFF_RATIO_RANGE)
load_eiz_based_looc_result_from_csv(EIZ_LAT_RANGE)

plot_metric_against_sat_off_ratio_and_orb_off_ratio(SAT_OFF_RATIO_RANGE, ORB_OFF_RATIO_RANGE, 140, "isl_num")
plot_metric_against_sat_off_ratio_and_orb_off_ratio(SAT_OFF_RATIO_RANGE, ORB_OFF_RATIO_RANGE, 140, "avg_conn_latency")
plot_metric_against_sat_off_ratio_and_orb_off_ratio(SAT_OFF_RATIO_RANGE, ORB_OFF_RATIO_RANGE, 140, "nores_ratio")
plot_metric_against_sat_off_ratio_and_orb_off_ratio(SAT_OFF_RATIO_RANGE, ORB_OFF_RATIO_RANGE, 140, "nores_ratio_without_zerohop")
plot_metric_against_sat_off_ratio_and_orb_off_ratio(SAT_OFF_RATIO_RANGE, ORB_OFF_RATIO_RANGE, 140, "pow_cons")
plot_metric_against_eiz_lat(EIZ_LAT_RANGE, ARR_RATE_RANGE, "avg_conn_latency")
plot_metric_against_eiz_lat(EIZ_LAT_RANGE, ARR_RATE_RANGE, "isl_num")
plot_metric_against_eiz_lat(EIZ_LAT_RANGE, ARR_RATE_RANGE, "pow_cons")
plot_metric_against_eiz_lat(EIZ_LAT_RANGE, ARR_RATE_RANGE, "nores_ratio")
