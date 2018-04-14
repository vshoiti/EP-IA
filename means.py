# -*- coding: utf-8 -*-

import sys, random
import cPickle as pickle
from math import sqrt

def unpickle(fileName):
    f = open(fileName, 'r')
    p = pickle.Unpickler(f)
    return p.load()
    
def listOfAllTokens(corpus):
    allTokens = []
    for document in corpus:
        allTokens = allTokens + list(set(corpus[document]))
    return allTokens
    
def kmeans(corpus, k):
    allTokens = listOfAllTokens(corpus)
    centroids = randomInit(allTokens, k)
    #while (True):
    for text in corpus:
        print "todo"
            
def dist(centroid, text):
    sum = 0
    print text
    for token in centroid:
        try:
            sum += (centroid[token] - text[token]) ** 2
        except KeyError:
            sum += (centroid[token] - 0) ** 2
    return sqrt(sum)

def randomInit(allTokens, k):
    centroids = []
    for i in range(0, k):
        centroid = {}
        for token in allTokens:
            centroid[token] = random.uniform(0,1)
        centroids.append(centroid)
    return centroids

readFile = sys.argv[1]
writeFile = sys.argv[2]
corpus = unpickle(readFile)

corpus2 = kmeans(corpus,3)
