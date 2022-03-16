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
        logging.info('Read the configuration file {}.'.format(path))
        return cfg

    def get_spine_layer(self, cfg: ConfigParser):
        if not cfg.has_section('spine'):
            logging.error("Configuration file does not have 'spine' section.")
            raise Exception("Section spine is not included in cfg {}!".format(cfg.sections()))
        if cfg['spine'].keys() < {'exist', 'idle'}:
            logging.error("Missing conditions in configuration file {}.".format(cfg['spine'].keys()))
            raise Exception("Check the configuration file.")
        spine_layer_switches = []
        for i in cfg['spine']['exist'].split(','):
            switch = self.get_switch(i)
            switch.nport = switch.nport[1]              # only give the southbound ports
            switch.bandwidth = switch.bandwidth[1]      # only give the southbound bandwidth
            switch.cost = 0                             # the cost of existed switch is equal to 0
            spine_layer_switches.append(switch)
        for j in cfg['spine']['idle'].split(','):
            switch = self.get_switch(j)
            switch.nport = switch.nport[1]              # only give the southbound ports
            switch.bandwidth = switch.bandwidth[1]      # only give the southbound bandwidth
            spine_layer_switches.append(switch)
        logging.info('Get the spine layer switches')
        return spine_layer_switches

    def get_leaf_layer(self, cfg: ConfigParser):
        if not cfg.has_section('leaf'):
            logging.error("Configuration file does not have 'leaf' section.")
            raise Exception("Section leaf is not included in cfg {}!".format(cfg.sections()))
        if cfg['leaf'].keys() < {'exist', 'idle'}:
            logging.error("Missing conditions in configuration file {}.".format(cfg['leaf'].keys()))
            raise Exception("Check the configuration file.")
        leaf_layer_switches = []
        for i in cfg['leaf']['exist'].split(','):
            switch = self.get_switch(i)
            switch.nport = switch.nport[0]              # only give the northbound ports
            switch.bandwidth = switch.bandwidth[0]      # only give the northbound bandwidth
            switch.cost = 0                             # the cost of existed switch is equal to 0
            leaf_layer_switches.append(switch)
        for j in cfg['leaf']['idle'].split(','):
            switch = self.get_switch(j)
            switch.nport = switch.nport[0]              # only give the northbound ports
            switch.bandwidth = switch.bandwidth[0]      # only give the northbound bandwidth
            leaf_layer_switches.append(switch)
        logging.info('Get leaf layer switches.')
        return leaf_layer_switches

    def get_speed_list(self, cfg: ConfigParser):
        if not cfg.has_section('speed'):
            logging.error("Configuration file does not have 'speed' section.")
            raise Exception("Section speed is not included in cfg {}!".format(cfg.sections()))
        speed_list = cfg['speed']['speed']
        logging.info('Get the list of speed {}'.format(speed_list))
        return speed_list.split(',')

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

    def get_switch(self, index: str, path='../config/model/switch.ini'):
        cfg = self.read_config(path)
        switch_types = list(cfg.sections())
        if index not in switch_types:
            logging.error('Can not locate the switch.')
            raise Exception("Wrong switch index {}".format(index))
        return DataModel.Switch(cfg[index])
