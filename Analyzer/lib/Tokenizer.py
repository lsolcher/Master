# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:11:24 2018

@author: Luc
"""

import nltk
import logging
import os.path


def tokenize_from_dir_to_tokens_per_document(dirpath):
    """
    Takes in a list of directories or a single directory, reads in the text of all .txt in these dir(s) and tokenizes them
    :param dirpath: a single directory or a list of directories
    :return:
    """

    logger = logging.getLogger('root')
    logger.info('start tokenization')

    tokens = {}
    if type(dirpath) is list:
        for path in dirpath:
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    text = open(os.path.join(path, file), encoding="utf8", errors='ignore').read()
                    tokens[file] = nltk.word_tokenize(text, language='german')
    else:
        for file in os.listdir(dirpath):
            if file.endswith(".txt"):
                text = open(os.path.join(dirpath, file), encoding="utf8", errors='ignore').read()
                tokens[file] = nltk.word_tokenize(text, language='german')

    return tokens

def tokenize_from_dir_to_tokens_per_source(dirpath):
    """
    Takes in a list of directories or a single directory, reads in the text of all .txt in these dir(s) and tokenizes them
    :param dirpath: a single directory or a list of directories
    :return:
    """

    logger = logging.getLogger('root')
    logger.info('start tokenization')

    tokens = {}
    text = ''
    if type(dirpath) is list:
        for path in dirpath:
            text = ''
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    text += open(os.path.join(path, file), encoding="utf8", errors='ignore').read()
            tokens[path] = nltk.word_tokenize(text, language='german')
    else:
        for file in os.listdir(dirpath):
            text = ''
            if file.endswith(".txt"):
                text += open(os.path.join(dirpath, file), encoding="utf8", errors='ignore').read()
            tokens[file] = nltk.word_tokenize(text, language='german')

    return tokens


def tokenize_sentences(sentences):

    logger = logging.getLogger('root')
    logger.info('start tokenization')
    if type(sentences) is dict:
        tokens = {}
        for key, item in sentences.items():
            tokens[key] = [nltk.word_tokenize(sentence, language='german') for sentence in item]
        return tokens
    else:
        return nltk.word_tokenize(sentences, language='german')






