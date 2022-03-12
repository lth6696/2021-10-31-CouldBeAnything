import Input

import logging.config


if __name__ == '__main__':
    logging.config.fileConfig('../config/log/config.ini')
    cfg = Input.InputImp.InputImp().read_config()
    Input.InputImp.InputImp().get_spine_layer(cfg)
    a = Input.InputImp.InputImp()
    a.get_transceivers()