import os
import sys
import logging
from configparser import ConfigParser

sys.path.append("..")

from .InputApi import Input
from Model import DataModel


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
        logging.info('Read the topology configuration file.')
        return cfg

    def get_spine_layer(self, cfg: ConfigParser):
        if not cfg.has_section('spine'):
            logging.error("Configuration file does not have 'spine' section.")
            raise Exception("Section spine is not included in cfg {}!".format(cfg.sections()))
        spine_layer_switch = cfg['spine']
        logging.info('Get the spine layer switches {}'.format(spine_layer_switch))
        return spine_layer_switch

    def get_leaf_layer(self, cfg: ConfigParser):
        if not cfg.has_section('leaf'):
            logging.error("Configuration file does not have 'leaf' section.")
            raise Exception("Section leaf is not included in cfg {}!".format(cfg.sections()))
        leaf_layer_switch = []
        for i in cfg['leaf']['switch']:
            leaf_layer_switch.append(self.get_switch(i))
        logging.info('Get the leaf layer switches {}'.format(leaf_layer_switch))
        return leaf_layer_switch

    def get_speed_list(self, cfg: ConfigParser):
        if not cfg.has_section('speed'):
            logging.error("Configuration file does not have 'speed' section.")
            raise Exception("Section speed is not included in cfg {}!".format(cfg.sections()))
        speed_list = cfg['speed']['speed']
        logging.info('Get the list of speed {}'.format(speed_list))
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
        transceivers = dict()
        for t in cfg.sections():
            transceivers[t] = DataModel.Transceiver(cfg[t])
        return transceivers

    def get_switch(self, index):
        pass
