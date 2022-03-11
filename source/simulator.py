import Input

import logging.config


if __name__ == '__main__':
    logging.config.fileConfig('../config/log/config.ini')
    Input.InputImp.InputImp().read_config()