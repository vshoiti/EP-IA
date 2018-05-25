 # -*- coding: utf-8 -*-

import os
import re
import sys
import pickle
import nltk
import operator
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from math import log10

# usage: python pre.py [in_filename] [representation_char] [stemmer_binary] [cut_binary] [out_filename]
# example: python pre.py reuters.txt b 1 0 reuters_bin_stem
# b: binary
# t: tf
# i: tf-idf

def main():
    # estagio de preparacao
    sourcepath = sys.argv[1]
    representation = sys.argv[2]
    stemmer = True if int(sys.argv[3]) == 1 else False
    cut = sys.argv[4]
    targetpath = sys.argv[5]
    corpus1 = loadCorpus(sourcepath)
    #print('corpus1 len(): ', len(corpus1))
    # estagio de pre-processamemento
    param_foldCase = True
    param_language = 'english'
    param_listOfStopWords = stopwords.words(param_language)
    param_stemmer = SnowballStemmer(param_language) if stemmer else None
    params = (param_foldCase, param_listOfStopWords, param_stemmer)
    corpus2 = processCorpus(corpus1, params)
    freq_list = token_frequency_list(corpus2) # lista que contem todos os tokens e sua frequencia
    #print len(freq_list)
    num_terms_to_remove = len(freq_list)/200 * int(cut) # removeremos tokens acima do 99.5 percentil
    #print num_terms_to_remove
    blacklist = black_list(freq_list, num_terms_to_remove)
    #blacklist.append('aaa')
    #print len(blacklist)
    # estagio de representacao
    if representation == 'b':
        corpus3 = representCorpus_binary(corpus2, blacklist)
    if representation == 't':
        corpus3 = representCorpus_tf(corpus2, blacklist)
    if representation == 'i':
        corpus3 = representCorpus_tf_idf(corpus2, blacklist)
    serialise(corpus3, targetpath)
    corpus4 = represent_matrix(corpus3)
    serialise(corpus4, targetpath+"_som")
    #print corpus4


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
        if (param_stemmer):
            listOfTokens = applyStemming(listOfTokens, param_stemmer)
        
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

def representCorpus_tf_idf(corpus, blacklist):
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
        dictOfTfScoredTokens = tf(corpus[document], blacklist)
        # computa um dicionario com o tf-idf score de cada par termo-documento
        corpus[document] = ({token: dictOfTfScoredTokens[token] * idfDict[token] for token in dictOfTfScoredTokens})
    return corpus

def representCorpus_binary(corpus, blacklist):
    newCorpus = {}
    for document in corpus:
        newCorpus[document] = binary(corpus[document], blacklist)
    return newCorpus

def representCorpus_tf(corpus, blacklist    ):
    newCorpus = {}
    for document in corpus:
        newCorpus[document] = tf(corpus[document], blacklist)
    return newCorpus

def tf(listOfTokens, blacklist):
    # cria um dicionario associando cada token com o numero de vezes
    # em que ele ocorre no documento (cujo conteudo eh listOfTokens)
    types = {}
    for token in listOfTokens:
        try:
            types[token] += 1
        except KeyError:
            types[token] = 1

    types = removeTokens(types, blacklist)
    return types
    
def binary(listOfTokens, blacklist):
    types = {}
    for token in listOfTokens:
        types[token] = 1

    types = removeTokens(types, blacklist)
    return types

def removeTokens(tokenDict, blacklist):
    for token in blacklist:
        try:
            del tokenDict[token]
        except KeyError:
            continue
    return tokenDict

def token_frequency_list(corpus):
    frequency_list = {}
    for text in corpus:
        frequency_list = add_dicts(tf(corpus[text], []), frequency_list)
    return frequency_list

def black_list(freq_list, num_terms_to_remove):
    blacklist = []
    for word in freq_list.keys(): # para cada palavra que aparece 1 vez
        if (freq_list[word] == 1): blacklist.append(word)

    # ordena decresc. a lista de freq. em uma lista de palavras 
    sorted_key_list = sorted(freq_list.items(), key=operator.itemgetter(1), reverse=True) # https://stackoverflow.com/a/613218
    for i in range(num_terms_to_remove): # adiciona as 'num_terms'-primeiras palavras a blacklist
        #print sorted_key_list[i][0]
        blacklist.append(sorted_key_list[i][0])
    return blacklist

def add_dicts(partial_dict, total_dict):
    for token in partial_dict:
        try:
            total_dict[token] += 1
        except KeyError:
            total_dict[token] = 1
    return total_dict

def represent_matrix(corpus):
    allTokens = []
    for document in corpus:
        allTokens = list(set(allTokens + corpus[document].keys()))    

    new_corpus = []
    new_corpus.append(allTokens)
    for text in corpus:
        document = []
        for token in allTokens:
            try:
                document.append(corpus[text][token])
            except KeyError:
                document.append(0)
        new_corpus.append(document)
    
    return new_corpus
            

def serialise(obj, name):
    f = open(name + '.pkl', 'wb')
    p = pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()

#nltk.download('stopwords')
main()
