# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:13:28 2018

@author: Luc
"""

TESTPATH = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/TEST/'

files = []
for file in os.listdir(TESTPATH):
    if file.endswith(".txt"):
        files.append(open(os.path.join(TESTPATH, file), encoding="utf8", errors='ignore').read())
