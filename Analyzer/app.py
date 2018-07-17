# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:13:28 2018

@author: Luc
"""

import log
import os.path
import math
import operator
import time
import re
import csv
import preprocessor
import numpy as np
from lib.RNN import rnn_keras
from lib.RNN import rnn_tensorflow
import pickle

def load_obj(name):
    with open('data/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_dirs(main_path):
    directory_list = list()
    dirs = [os.path.join(main_path, d) for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
    for train_dir in dirs:
        directory_list.append(train_dir)
    return directory_list

def bow(main_path):
    directory_list = list()
    dirs = [os.path.join(main_path, d) for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
    for train_dir in dirs:
        directory_list.append(train_dir)
    counts, word_list = preprocessor.preprocess_tokens_per_document(directory_list)

    #BOW
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




def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


def prepare_data(main_path, per_source, test_string):
    if per_source is False:
        if os.path.isfile('data/obj/counts'+test_string+'.pkl'):
            counts = load_obj('counts'+test_string)
            word_list = load_obj('word_list'+test_string)
            tokens = load_obj('tokens'+test_string)
        else:
            counts, word_list, tokens = preprocessor.preprocess_tokens_per_document(main_path, test_string)
    else:
        if os.path.isfile('data/obj/counts_source'+test_string+'.pkl'):
            counts = load_obj('counts_source'+test_string)
            word_list = load_obj('word_list_source'+test_string)
            tokens = load_obj('tokens_source'+test_string)

        else:
            counts, word_list, tokens = preprocessor.preprocess_tokens_per_source(main_path, test_string)


    return counts, word_list, tokens


def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))


def main():
    # logger
    per_source = True
    datapath = os.path.abspath(os.path.dirname(__file__)) + '\\data\\'
    logger = log.setup_custom_logger('root')
    logger.info('start analyzing')
    main_path = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/'
    test_path = 'C:/Programmierung/Masterarbeit/Scraper/data/test'
    valid_path = 'C:/Programmierung/Masterarbeit/Scraper/data/valid'
    model_path = 'C:/Programmierung/Masterarbeit/Analyzer/data/trainedModels'

    #dirs_to_train = load_dirs(main_path)

    counts, word_list, tokens = prepare_data(main_path, per_source, '')
    counts_test, word_list_test, tokens_test = prepare_data(test_path, per_source, '_test')
    counts_valid, word_list_valid, tokens_valid = prepare_data(valid_path, per_source, '_valid')
    if os.path.isfile('data/obj/articles.pkl'):
        articles = load_obj('articles')
    else:
        articles = preprocessor.get_articles_from_top_dir(main_path, '')
    if os.path.isfile('data/obj/articles_test.pkl'):
        articles_test = load_obj('articles_test')
    else:
        articles_test = preprocessor.get_articles_from_top_dir(test_path, '_test')


    # cnn_model.train()
    rnn_tensorflow.run(articles, articles_test)
    #rnn_keras.run(articles, articles_test)



    # rnn_lstm_2.load_data(model_path, counts, word_list, tokens, articles, counts_test, word_list_test, tokens_test, counts_valid, word_list_valid, tokens_valid)
    # word2vec.train(counts, word_list, tokens, articles)
    # counts, word_list, tokens = preprocessor.preprocess_tokens_per_document(main_path)

    # rnn_lstm.prepare_data(counts, word_list, tokens, articles)
    # rnn_udemy.prepare_data(counts, word_list, tokens, articles)
    #bow(main_path)
    #rnn_word_model.load_data(main_path)
    #rnn_char_model.main(main_path)
    #bow(main_path)
    #rnn.train_rnn(main_path)
    #model = RNNTheano(8000, hidden_dim=100)
    #load_model_parameters_theano("C:\\Programmierung\\Masterarbeit\\Analyzer\\data\\trainedModels\\rnn-theano-80-8000-2018-06-19-07-10-30.npz", model)
    #rnn.rnn(main_path, model)


if __name__ == "__main__":
    main()



