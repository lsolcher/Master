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
import math


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

TESTPATH = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/TEST/'
tokens = Tokenizer.tokenize_from_dir_to_tokens_per_document(TESTPATH)
tokens = Tagger.tag(tokens)
tokens = Normalizer.normalize(tokens)
tokens = Lemmatizer.lemmatize_tokens(tokens)

word_list = []
for token in tokens:
    word_list.append([t[0] for t in token])
counts = []
for l in word_list:
    counts.append(Counter(l))

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

doc_names = [f for f in listdir(TESTPATH) if isfile(join(TESTPATH, f))]
print(doc_names)

#raw_counts = np.mat(bow, dtype=float)
#tf = raw_counts / np.sum(raw_counts, axis=1)

#sims = gensim.similarities.Similarity()

sim1 = counter_cosine_similarity(counts[0], counts[1])
sim2 = counter_cosine_similarity(counts[0], counts[2])
sim3 = counter_cosine_similarity(counts[1], counts[2])

print(sim1)
print(sim2)
print(sim3)



