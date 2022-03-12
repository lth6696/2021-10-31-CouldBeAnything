import os
import logging
from configparser import ConfigParser

from .InputApi import Input


class InputImp(Input):
    def __init__(self):
        Input.__init__(self)

    def read_config(self, path=''):
        if not path:
            path = '../config/small/config.ini'
        if not os.path.exists(path):
            logging.error("Configuration file does not exist.")
            raise Exception("Invalid path {}!".format(path))
        cfg = ConfigParser()
        cfg.read(path)
        return cfg

    def get_spine_layer(self, cfg: ConfigParser):
        if not cfg.has_section('spine'):
            logging.error("Configuration file does not have 'spine' section.")
            raise Exception("Section spine is not included in cfg {}!".format(cfg.sections()))
        spine_layer_switch = cfg['spine']
        return spine_layer_switch

    def get_leaf_layer(self, cfg: ConfigParser):
        if not cfg.has_section('leaf'):
            logging.error("Configuration file does not have 'leaf' section.")
            raise Exception("Section leaf is not included in cfg {}!".format(cfg.sections()))
        leaf_layer_switch = cfg['leaf']
        return leaf_layer_switch

    def get_speed_list(self, cfg: ConfigParser):
        if not cfg.has_section('speed'):
            logging.error("Configuration file does not have 'speed' section.")
            raise Exception("Section speed is not included in cfg {}!".format(cfg.sections()))
        speed_list = cfg['speed']
        return speed_list

    def get_transceivers(self, cfg: ConfigParser = None, path=''):
        if cfg:
           pass
        elif not path:
            path = '../config/model/transceiver.ini'
            cfg = self.read_config(path)
        else:
            logging.error("Can not acquire transceiver info.")
            raise Exception("No input found.")
        return dict(cfg)
