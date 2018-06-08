# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:11:24 2018

@author: Luc
"""

import nltk
import logging
import os.path

def tokenize(dirpath):
    
    logger = logging.getLogger('root')
    logger.info('start tokenization')
    TESTPATH = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/TEST/'   
    
    files = []
    tokens = []
    for file in os.listdir(dirpath):
        if file.endswith(".txt"):
            files.append(open(os.path.join(TESTPATH, file), encoding="utf8", errors='ignore').read());
    
    for idx, file in enumerate(files):
        tokens.append(nltk.word_tokenize(file, language='german'))
                
    ###file = open('C:/Programmierung/Masterarbeit/Scraper/data/articles/TEST/spon.txt', encoding='utf-8', errors='ignore').read();
        
    #resultfile = open(os.path.join(RESULTPATH, 'result.txt'), 'w')
    #for item in tokens[1]:
     # resultfile.write("%s\n" % item)
     
    return tokens;
        
    
        
    
          
