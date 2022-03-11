import configparser


class Input:
    def __init__(self):
        pass

    def read_config(self, path=''):
        # Read the configuration file and store it in memory.
        return configparser.ConfigParser

    def get_spine_layer(self, cfg: configparser.ConfigParser):
        # Read the info of spine-layer switches from cfg.
        return dict

    def get_leaf_layer(self, cfg: configparser.ConfigParser):
        # Read the info of leaf-layer switches from cfg.
        return dict

    def get_speed_list(self, cfg: configparser.ConfigParser):
        # Acquire the set of transmission rate that may be used.
        return list

    def get_transceivers(self, cfg: configparser.ConfigParser):
        # Acquire the set of optical modules.
        return dict