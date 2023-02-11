#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging

def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)#正常WARNING,INFO
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

# import logging
# log_format = "%(asctime)s -%(levelname)s- %(message)s"
# date_format = "%m/%d/%Y %H:%M:%S %p"
# logging.basicConfig(filename='./log/my.log',level=logging.INFO,format=log_format, datefmt=date_format)
