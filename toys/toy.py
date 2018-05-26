# -*- coding: utf-8 -*-

import os
import sys
import pickle
import re

# script for turning toy .txt datasets into .pkls for testing

def loadCorpus(sourcepath):
    corpus = {}
    with open(sourcepath, 'r') as f:
        for line in f:
            if line.strip(): # se a line nao for vazia, line.strip() == true
                corpus[line] = line
    return corpus

def serialise(obj, name):
    f = open(name + '.pkl', 'wb')
    p = pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()

def representCorpus(corpus):
    newCorpus = {}
    for text in corpus: # cada linha em corpus representa um texto, itera pelas linhas
        text = text.rstrip('\n')
        regExpr = ',+'
        tokens = filter(None, re.split(regExpr, text))
        values = {}
        for i in range(len(tokens)):
            try:
                values[i] = float(tokens[i])
            except ValueError:
                pass # tokens[i] is a label
        newCorpus[text] = values

    return newCorpus

def main():
    # estagio de preparacao
    sourcepath = sys.argv[1]
    targetpath = sys.argv[2]
    corpus1 = loadCorpus(sourcepath)
    corpus2 = representCorpus(corpus1)
    serialise(corpus2, targetpath)
    print(corpus2)


main()
