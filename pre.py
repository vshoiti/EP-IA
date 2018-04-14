# -*- coding: utf-8 -*-

import os
import re
import sys
import cPickle as pickle
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from math import log10

def main():
    # estagio de preparacao
    sourcepath = sys.argv[1]
    targetpath = sys.argv[2]
    corpus1 = loadCorpus(sourcepath)
    # estagio de pre-processamemento
    param_foldCase = True
    param_language = 'english'
    param_listOfStopWords = stopwords.words(param_language)
    param_stemmer = SnowballStemmer(param_language)
    params = (param_foldCase, param_listOfStopWords, param_stemmer)
    corpus2 = processCorpus(corpus1, params)
    # estagio de representacao
    # remova o '#' de uma das linhas abaixo 
    #corpus3 = representCorpus_binary(corpus2)
    #corpus3 = representCorpus_tf(corpus2)
    #corpus3 = representCorpus_tf_idf(corpus2)
    serialise(corpus3, targetpath)

def loadCorpus(sourcepath):
    corpus = {}
    with open(sourcepath, 'r') as f:
        for line in f:
        	if line.strip(): # se a line nao for vazia, line.strip() == true
	            corpus[line] = line
    return corpus

def processCorpus(corpus, params):
    (param_foldCase, param_listOfStopWords, param_stemmer) = params
    newCorpus = {}
    for text in corpus: # cada linha em corpus representa um texto, itera pelas linhas
        text = text.rstrip('\n')
        text = foldCase(text, param_foldCase)
        listOfTokens = tokenize(text)
        listOfTokens = removeStopWords(listOfTokens, param_listOfStopWords)
        #listOfTokens = applyStemming(listOfTokens, param_stemmer)
        newCorpus[text] = listOfTokens
    return newCorpus

def foldCase(sentence, parameter):
    if(parameter): sentence = sentence.lower()
    return sentence

def tokenize(sentence):
    sentence = sentence.replace("_"," ")
    regExpr = '\W+'
    return filter(None, re.split(regExpr, sentence))

def removeStopWords(listOfTokens, listOfStopWords):
    return [token for token in listOfTokens if token not in listOfStopWords]

def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

def representCorpus_tf_idf(corpus):
    # cria uma lista com todos os tokens distintos que ocorrem em cada documento.
    allTokens = []
    for document in corpus:
        allTokens = allTokens + list(set(corpus[document]))

    # cria o dicionario reverso
    idfDict = {}
    for token in allTokens:
        try:
            idfDict[token] += 1
        except KeyError:
            idfDict[token] = 1

    # atualiza o dicionario reverso, associando cada token com seu idf score
    nDocuments = len(corpus)
    for token in idfDict:
        idfDict[token] = log10(nDocuments/float(idfDict[token]))

    # computa a matriz termo-documento (newCorpus)
    for document in corpus:
        # computa um dicionario com os tf scores de cada termo que ocorre no documento
        dictOfTfScoredTokens = tf(corpus[document])
        # computa um dicionario com o tf-idf score de cada par termo-documento
        corpus[document] = ({token: dictOfTfScoredTokens[token] * idfDict[token] for token in dictOfTfScoredTokens})
    return corpus

def representCorpus_binary(corpus):
	newCorpus = {}
	for document in corpus:
		newCorpus[document] = binary(corpus[document])
	return newCorpus

def representCorpus_tf(corpus):
	newCorpus = {}
	for document in corpus:
		newCorpus[document] = tf(corpus[document])
	return newCorpus

def tf(listOfTokens):
    # cria um dicionario associando cada token com o numero de vezes
    # em que ele ocorre no documento (cujo conteudo eh listOfTokens)
    types = {}
    for token in listOfTokens:
        if(token in types.keys()): types[token] += 1
        else: types[token] = 1
    return types
    
def binary(listOfTokens):
    types = {}
    for token in listOfTokens:
        types[token] = 1
    return types
	
def serialise(obj, name):
    f = open(name + '.pkl', 'wb')
    p = pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()

nltk.download('stopwords')
main()
