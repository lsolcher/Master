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
import re
import csv


def bow(preprocessed_tokens):
    start = time.time()
    word_list = {}
    for idx, token in preprocessed_tokens.items():
        word_list[idx] = ([t[0] for t in token])
    counts = {}
    print('counting...')
    for idx, item in word_list.items():
        counts[idx] = Counter(item)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    vocab = []
    for l in word_list:
        vocab.append(set())

    # for i, c in enumerate(counts):
    #   vocab[i] |= set(c.keys())

    # for v in vocab:
    #   v = sorted(list(v))  # sorting here only for better display later

    # bow = []
    # for i, counter in enumerate(counts):
    #   bow_row = [counter.get(term, 0) for term in vocab[i]]
    #  bow.append(bow_row)

    print('calculating similarities...')
    similarities = {}
    for i, c in counts.items():
        for j, c1 in counts.items():
            author1 = " ".join(re.findall("[a-zA-Z]+", i))
            author2 = " ".join(re.findall("[a-zA-Z]+", j))
            if author1 != author2 and i < j:
                similarities[i + ' : ' + j] = counter_cosine_similarity(counts[i], counts[j])

    sorted_sims = sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)
    end = time.time()
    print('done! took ', end - start, ' seconds.')

    resultfile = 'C:/Programmierung/Masterarbeit/Analyzer/results/results.csv'

    with open(resultfile, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerows(sorted_sims)


def preprocess_tokens(dir):
    tokens = {}
    paths = []
    start = time.time()
    print('tokenizing...')
    for item in dir:
        paths.append(item)
    tokens = Tokenizer.tokenize(paths)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()

    print('tagging...')
    tokens = Tagger.tag(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('normalizing...')
    tokens = Normalizer.normalize(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    start = time.time()
    print('lemmatizing...')
    tokens = Lemmatizer.lemmatize(tokens)
    end = time.time()
    print('done! took ', end - start, ' seconds.')
    return tokens


def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


def main():
    # logger
    datapath = os.path.abspath(os.path.dirname(__file__)) + '\\data\\'
    logger = log.setup_custom_logger('root')
    logger.info('start analyzing')

    main_path = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/'
    directory_list = list()
    for root, dirs, files in os.walk(main_path, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))
    the_tokens = preprocess_tokens(directory_list)

    #BOW
    bow(the_tokens)


if __name__ == "__main__":
    main()



