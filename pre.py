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

# usage: python pre.py [in_filename] [representation_char] [stemmer_binary] [upper_cut_float] [lower_cut_float] [out_filename]
# example: python pre.py reuters.txt b 1 0.05 0.05 reuters_bin_stem_u5_l5
# b: binary
# t: tf
# i: tf-idf


def main():
    # todo: comment
    # estagio de preparacao
    sourcepath = sys.argv[1]
    representation = sys.argv[2]
    stemmer = True if int(sys.argv[3]) == 1 else False
    upper_cut_percentage = sys.argv[4]
    lower_cut_percentage = sys.argv[5]
    targetpath = sys.argv[6]
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
    #print(freq_list)
    #print(len(freq_list))
    upper_terms_to_remove = int(len(freq_list) * float(upper_cut_percentage))
    lower_terms_to_remove = int(len(freq_list) * float(lower_cut_percentage))
    #print upper_terms_to_remove
    blacklist = black_list(freq_list, upper_terms_to_remove, lower_terms_to_remove)
    #print(len(blacklist))
    #blacklist.append('aaa')
    #for token in blacklist:
    #    print(token, freq_list[token])
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


# todo: comment
def loadCorpus(sourcepath):
    corpus = {}
    with open(sourcepath, 'r') as f:
        for line in f:
            if line.strip(): # se a line nao for vazia, line.strip() == true
                corpus[line] = line
    return corpus


# todo: comment
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


# todo: comment
def foldCase(sentence, parameter):
    if(parameter): sentence = sentence.lower()
    return sentence


# todo: comment
def tokenize(sentence):
    sentence = sentence.replace("_"," ")
    regExpr = '\W+'
    return filter(None, re.split(regExpr, sentence))


# todo: comment
def removeStopWords(listOfTokens, listOfStopWords):
    return [token for token in listOfTokens if token not in listOfStopWords]


# todo: comment
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]


# todo: comment
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


# todo: comment
def representCorpus_binary(corpus, blacklist):
    newCorpus = {}
    for document in corpus:
        newCorpus[document] = binary(corpus[document], blacklist)
    return newCorpus


# todo: comment
def representCorpus_tf(corpus, blacklist    ):
    newCorpus = {}
    for document in corpus:
        newCorpus[document] = tf(corpus[document], blacklist)
    return newCorpus


# todo: comment
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


# todo: comment
def binary(listOfTokens, blacklist):
    types = {}
    for token in listOfTokens:
        types[token] = 1

    types = removeTokens(types, blacklist)
    return types


# todo: comment
def removeTokens(tokenDict, blacklist):
    for token in blacklist:
        try:
            del tokenDict[token]
        except KeyError:
            continue
    return tokenDict


# todo: comment
def token_frequency_list(corpus):
    frequency_list = {}
    for text in corpus:
        frequency_list = add_dicts(tf(corpus[text], []), frequency_list)
    return frequency_list


# todo: comment
def black_list(freq_list, upper_terms_to_remove, lower_terms_to_remove):
    blacklist = []
    for word in freq_list.keys(): # para cada palavra que aparece 1 vez
        if freq_list[word] == 1:
            blacklist.append(word) # adiciona o termo a blacklist

    for word in blacklist:
        del freq_list[word]  # remove o termo da lista para nao ser removido novamente abaixo

    # ordena crescentemente a lista de freq. em uma lista de palavras
    sorted_key_list = sorted(freq_list.items(), key=operator.itemgetter(1))  # https://stackoverflow.com/a/613218
    for i in range(lower_terms_to_remove):
        blacklist.append(sorted_key_list[i][0])

    sorted_key_list = list(reversed(sorted_key_list))
    for i in range(upper_terms_to_remove):
        blacklist.append(sorted_key_list[i][0])
    return blacklist


# todo: comment
def add_dicts(partial_dict, total_dict):
    for token in partial_dict:
        try:
            total_dict[token] += 1
        except KeyError:
            total_dict[token] = 1
    return total_dict


# todo: comment
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


# todo: comment
def serialise(obj, name):
    f = open(name + '.pkl', 'wb')
    p = pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()


if __name__ == '__main__':
    #nltk.download('stopwords')
    main()
