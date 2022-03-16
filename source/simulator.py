from Input import InputImp
from Algorithm import AlgorithmImp

import logging.config


def activate():
    cfg = InputImp.InputImp().read_config()     # read topology configuration
    input = InputImp.InputImp()
    leaf_layer_switches = input.get_leaf_layer(cfg)
    spine_layer_switches = input.get_spine_layer(cfg)
    speed = [int(k) for k in input.get_speed_list(cfg)]

    algorithm = AlgorithmImp.AlgorithmImp()
    ilp = algorithm.linear_integer_programming(spine_layer_switches, leaf_layer_switches,speed, 1200)


if __name__ == '__main__':
    logging.config.fileConfig('../config/log/config.ini')
    activate()