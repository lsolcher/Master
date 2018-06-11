# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:24:10 2018

@author: Luc
"""
import nltk


def normalize(tokens):
    
    print(len(tokens))

    for t in tokens:
        print(len(t))
        t = remove_special_chars(t)
        print(len(t))
        t = remove_stopwords(t)
        print(len(t))
        t = stem(t)
        print(len(t))
        # correct spelling

    print("done")
    return tokens

    
def remove_stopwords(token):

    new_token = []
    stopwords = nltk.corpus.stopwords.words('german')
    for t in token:
        if t.lower() not in stopwords:
            new_token.append(t)
    return new_token


def remove_special_chars(token):

    new_token = []
    for t in token:
        if t.isalpha():
            new_token.append(t)
    return new_token


def stem(token):

    stemmer = nltk.stem.SnowballStemmer('german')
    new_token = []
    for t in token:
        new_token.append(stemmer.stem(t))
        #print(t)
    return new_token

            

