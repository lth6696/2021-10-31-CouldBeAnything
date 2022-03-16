import logging
import configparser


class Transceiver:
    def __init__(self, *args, **kwargs):
        self.name = ""
        self.speed = 0
        self.cost = 0

        self._init_values(*args, **kwargs)

    def _init_values(self, *args, **kwargs):
        if len(args) and type(args[0]) == configparser.SectionProxy:
            self.name = str(args[0]['name'])
            self.speed = int(float(args[0]['speed']))
            self.cost = int(float(args[0]['cost']))
            logging.info('Initialize transceiver {}'.format(self.name))
        elif 'section' in kwargs.keys():
            self.name = str(kwargs['section']['name'])
            self.speed = int(float(kwargs['section']['speed']))
            self.cost = int(float(kwargs['section']['cost']))
            logging.info('Initialize transceiver {}'.format(self.name))
        elif kwargs.keys() >= {'name', 'speed', 'cost'}:
            self.name = str(kwargs['name'])
            self.speed = int(float(kwargs['speed']))
            self.cost = int(float(kwargs['cost']))
            logging.info('Initialize transceiver {}'.format(self.name))
        else:
            logging.error("Failed to initialize a transceiver.")
            return None


class Switch:
    def __init__(self, *args, **kwargs):
        self.name = ''
        self.nport = ()
        self.bandwidth = ()     # Gbps
        self.cost = 0
        self.type = bool

        self._init_values(*args, **kwargs)

    def _init_values(self, *args, **kwargs):
        if len(args) and type(args[0]) == configparser.SectionProxy:
            self.name = str(args[0]['name'])
            self.nport = (int(args[0]['uport']), int(args[0]['dport']))
            self.bandwidth = (int(float(args[0]['ubandwidth'])), int(float(args[0]['dbandwidth'])))
            self.cost = int(args[0]['cost'])
            self.type = 1 if args[0]['type'] == 'EPS' else 0
            logging.info('Initialize switch {}'.format(self.name))
        elif 'section' in kwargs.keys():
            self.name = str(kwargs['section']['name'])
            self.nport = (int(float(kwargs['section']['uport'])),
                          int(float(kwargs['section']['dport'])))
            self.bandwidth = (int(float(kwargs['section']['ubandwidth'])),
                              int(float(kwargs['section']['dbandwidth'])))
            self.cost = int(float(kwargs['section']['cost']))
            self.type = 1 if kwargs['section']['type'] == 'EPS' else 0
            logging.info('Initialize switch {}'.format(self.name))
        elif kwargs.keys() >= {'name', 'uport', 'dport', 'ubandwidth', 'dbandwidth', 'cost', 'type'}:
            self.name = str(kwargs['name'])
            self.nport = (int(float(kwargs['uport'])), int(float(kwargs['dport'])))
            self.bandwidth = (int(float(kwargs['ubandwidth'])), int(float(kwargs['dbandwidth'])))
            self.cost = int(float(kwargs['cost']))
            self.type = 1 if kwargs['type'] == 'EPS' else 0
            logging.info('Initialize switch {}'.format(self.name))
        else:
            logging.error("Failed to initialize a switch.")
            return None
