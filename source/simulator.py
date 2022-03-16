from Input import InputImp

import logging.config


def activate():
    cfg = InputImp.InputImp().read_config()
    InputImp.InputImp().get_leaf_layer(cfg)
    InputImp.InputImp().get_spine_layer(cfg)
    a = InputImp.InputImp()


if __name__ == '__main__':
    logging.config.fileConfig('../config/log/config.ini')
    activate()