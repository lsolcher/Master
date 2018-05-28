# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:44:00 2018

@author: Luc
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy

SPONDIRPATH = 'C:/Programmierung/Masterarbeit/Scraper/SPON/articles/'

files = [open(SPONDIRPATH + 'SPON_0.txt', 'r'), open(SPONDIRPATH + 'SPON_1.txt', 'r'), open(SPONDIRPATH + 'testfile.txt', 'r')]

vectorizer = CountVectorizer(input='file')
dtm = vectorizer.fit_transform(files)
vocab = vectorizer.get_feature_names()
scipy.sparse.csr.csr_matrix
dtm = dtm.toarray()
vocab = np.array(vocab)
house_idx = list(vocab).index('aber')
dtm[0, vocab == 'house']
n, _ = dtm.shape
dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        x, y = dtm[i, :], dtm[j, :]
        dist[i, j] = np.sqrt(np.sum((x - y)**2))
dist = euclidean_distances(dtm)
np.round(dist, 1)
print(dist[0, 2])

"""
file = open(SPONDIRPATH + 'SPON_0.txt', 'r') 
text = file.read().split(',')

# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
dtm = vectorizer.fit_transform(text)
vocab = vectorizer.get_feature_names()
print(dtm)
print(vocab)
dtm = dtm.toarray()
vocab = np.array(vocab)
"""
"""
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())




result = open('C:/Programmierung/Masterarbeit/Analyzer/results/result_0.txt', 'w')
pickle.dump(vectorizer.get_feature_names(), result)

#result.write(vectorizer.vocabulary_)
#result.write(vector.shape)
#result.write(type(vector))
#result.write(vector.toarray()
"""