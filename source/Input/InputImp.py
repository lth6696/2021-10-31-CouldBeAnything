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

    def get_speed_list(self, cfg: ConfigParser):

        return None
