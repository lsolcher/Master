# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:44:00 2018

@author: Luc
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import os
from os.path import isfile, join
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

DIRPATH = 'C:/Programmierung/Masterarbeit/Scraper/data/articles/'
RESULTPATH = 'C:/Programmierung/Masterarbeit/Analyzer/results/'

SPONPATH = os.path.join(DIRPATH, 'SPON')
ZEITPATH = os.path.join(DIRPATH, 'ZEIT')

sponFiles = []
for file in os.listdir(SPONPATH):
    if file.endswith(".txt"):
        sponFiles.append(open(os.path.join(SPONPATH, file), encoding="utf8", errors='ignore').read())
zeitFiles = []
for file in os.listdir(ZEITPATH):
    if file.endswith(".txt"):
        zeitFiles.append(open(os.path.join(ZEITPATH, file), encoding="utf8", errors='ignore').read())


#files = [open(SPONDIRPATH + 'SPON_0.txt', 'r'), open(SPONDIRPATH + 'SPON_1.txt', 'r'), open(SPONDIRPATH + 'testfile.txt', 'r')]
files = sponFiles + zeitFiles
print(len(sponFiles))
print(len(zeitFiles))
print(len(files))



vectorizer = CountVectorizer(input='files')
dtm = vectorizer.fit_transform(files)
vocab = vectorizer.get_feature_names()
scipy.sparse.csr.csr_matrix
dtm = dtm.toarray()
vocab = np.array(vocab)
n, _ = dtm.shape
dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        x, y = dtm[i, :], dtm[j, :]
        dist[i, j] = np.sqrt(np.sum((x - y)**2))
dist = euclidean_distances(dtm)
#resultfile = open(os.path.join(RESULTPATH, 'result.csv'), 'wb')
#resultfile.write(dtm)
print(dtm)
np.round(dist, 1)
print(dist[0, 2])
min_dist = 1000
for i in range(438):
    for j in range(48):
        print(i)
        print(j)
        thisDist = dist[i, 438 + j]
        print(thisDist)
        if min_dist > thisDist:
            print('new min dist:')
            print(min_dist)
            min_dist = thisDist
            sponA = i
            zeitA = j
print(sponA)
print(zeitA)
print()


# Visualization
"""
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]
names = [os.path.basename(fn).replace('.txt', '') for fn in files]
print('show plot')
for x, y, name in zip(xs, ys, names):
    color = 'orange' if "Austen" in name else 'skyblue'
    plt.scatter(x, y, c=color)
    plt.text(x, y, name)
plt.show()
"""
#scipy.spatial.distance.pdist(dist)
#print(scipy.spatial.distance.pdist(dist))


"""
vectorizerSpon = CountVectorizer(input='sponFiles')
dtmSpon = vectorizerSpon.fit_transform(sponFiles)
vocab = vectorizerSpon.get_feature_names()
scipy.sparse.csr.csr_matrix
dtmSpon = dtmSpon.toarray()
vocab = np.array(vocab)
n, _ = dtmSpon.shape
distSpon = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        x, y = dtmSpon[i, :], dtmSpon[j, :]
        distSpon[i, j] = np.sqrt(np.sum((x - y)**2))
np.round(distSpon, 1)

        
vectorizerZeit = CountVectorizer(input='zeitFiles')
dtmZeit = vectorizerZeit.fit_transform(zeitFiles)
vocab = vectorizerZeit.get_feature_names()
scipy.sparse.csr.csr_matrix
dtmZeit = dtmZeit.toarray()
vocab = np.array(vocab)
n, _ = dtmZeit.shape
dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        x, y = dtmZeit[i, :], dtmZeit[j, :]
        dist[i, j] = np.sqrt(np.sum((x - y)**2))
distZeit = euclidean_distances(dtmZeit)


#resultfile = open(os.path.join(RESULTPATH, 'result.csv'), 'wb')
#resultfile.write(dtm)
#print(dtm)
print(dist[0, 2])
"""

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