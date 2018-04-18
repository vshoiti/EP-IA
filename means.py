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
    for n in (range(10)):
        centroidWins = [[] for x in xrange(k)]
        for text in corpus:
            closestCentroid = findBMU(centroids, corpus[text], k)
            centroidWins[closestCentroid].append(corpus[text])
	print centroidWins
        newCentroidPositions(centroids, centroidWins)
    print centroids

def newCentroidPositions(centroids, centroidWins):
    for i in range(len(centroids)):
        centroid = centroids[i]
        closestTexts = centroidWins[i]
        numOfWins = len(closestTexts) if len(closestTexts) > 1 else 1
        for token in centroid:
            newTokenValueSum = centroid[token]
            for text in closestTexts:
                try:
                    newTokenValueSum += text[token]
                except KeyError: # o texto nao contem aquele token
                    newTokenValueSum += 0
            centroid[token] = newTokenValueSum / numOfWins

def findBMU(centroids, text, k):
    minDistance = -1
    closestCentroid = -1
    for i in range(k):
        currentDist = dist(centroids[i], text)
        if currentDist < minDistance or minDistance == -1:
            minDistance = currentDist
            closestCentroid = i
    return closestCentroid
            
def dist(centroid, text):
    sum = 0
    for token in centroid:
        try:
            sum += (centroid[token] - text[token]) ** 2
        except KeyError: # o texto nao contem aquele token
            sum += (centroid[token] - 0) ** 2
    return sqrt(sum)

def randomInit(allTokens, k):
    centroids = []
    for i in range(k):
        centroid = {}
        for token in allTokens:
            centroid[token] = random.uniform(0,1)
        centroids.append(centroid)
    return centroids

readFile = sys.argv[1]
writeFile = sys.argv[2]
corpus = unpickle(readFile)
corpus2 = kmeans(corpus,3)
