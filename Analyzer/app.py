# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:13:28 2018

@author: Luc
"""

import log
from os import listdir
import os.path
from os.path import isfile, join
from lib import Tokenizer, Normalizer, Tagger, Lemmatizer
from collections import Counter
import numpy as np
import gensim
import math
import operator
import time



def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

# logger
datapath = os.path.abspath(os.path.dirname(__file__)) + '\\data\\'
logger = log.setup_custom_logger('root')
logger.info('start analyzing')

main_path = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/'
directory_list = list()
for root, dirs, files in os.walk(main_path, topdown=False):
    for name in dirs:
        directory_list.append(os.path.join(root, name))
tokens = []
start = time.time()
print('tokenizing...')
for item in directory_list:
    tokens.append(Tokenizer.tokenize(item))
end = time.time()
print('done! took ' + end-start + ' seconds.')
start = time.time()
print('tagging...')
tokens = Tagger.tag(tokens)
end = time.time()
print('done! took ' + end-start + ' seconds.')
start = time.time()
print('normalizing...')
tokens = Normalizer.normalize(tokens)
end = time.time()
print('done! took ' + end-start + ' seconds.')
start = time.time()
print('lemmatizing...')
tokens = Lemmatizer.lemmatize(tokens)
end = time.time()
print('done! took ' + end-start + ' seconds.')
start = time.time()
word_list = []
for token in tokens:
    word_list.append([t[0] for t in token])
counts = []
print('counting...')
for l in word_list:
    counts.append(Counter(l))
end = time.time()
print('done! took ' + end-start + ' seconds.')
start = time.time()
vocab = []
for l in word_list:
    vocab.append(set())

for i, c in enumerate(counts):
    vocab[i] |= set(c.keys())

for v in vocab:
    v = sorted(list(v))  # sorting here only for better display later
    print(v)   # => becomes columns of BoW matrix

bow = []
for i, counter in enumerate(counts):
    bow_row = [counter.get(term, 0) for term in vocab[i]]
    bow.append(bow_row)

print('calculating similarities...')
similarities = {}
for i, c in enumerate(counts):
    for j, c1 in enumerate(counts):
        if i != j:
            similarities[i+':'+j] = counter_cosine_similarity(counts[i], counts[j])

sorted_sims = sorted(similarities.items(), key=operator.itemgetter(0), reverse=True)
end = time.time()
print('done! took ' + end-start + ' seconds.')

print(sorted_sims)



