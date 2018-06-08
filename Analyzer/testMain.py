# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:13:28 2018

@author: Luc
"""

import log
import os.path

# logger
datapath = os.path.abspath(os.path.dirname(__file__)) + '\\data\\'
logger = log.setup_custom_logger('root')
logger.info('start analyzing')

TESTPATH = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/TEST/'

files = []
for file in os.listdir(TESTPATH):
    if file.endswith(".txt"):
        files.append(open(os.path.join(TESTPATH, file), encoding="utf8", errors='ignore').read())
