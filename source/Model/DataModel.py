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
        elif kwargs.keys() >= {'name', 'nport', 'bandwidth', 'cost', 'type'}:
            self.name = str(kwargs['name'])
            self.speed = int(float(kwargs['speed']))
            self.cost = int(float(kwargs['cost']))
            logging.info('Initialize transceiver {}'.format(self.name))
        else:
            logging.error("Failed to initialize a transceiver.")
            return None


class Switch:
    def __init__(self):
        self.name = ''
        self.nport = ()
        self.bandwidth = ()
        self.cost = 0
        self.type = bool

        self._init_values()

    def _init_values(self):
        pass