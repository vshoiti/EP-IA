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
    # estagio de pre-processamemento
    param_foldCase = True
    param_language = 'english'
    param_listOfStopWords = stopwords.words(param_language)  # recebe uma lista de stopwords inglesas do nltk
    param_stemmer = SnowballStemmer(param_language) if stemmer else None
    params = (param_foldCase, param_listOfStopWords, param_stemmer)
    corpus2 = processCorpus(corpus1, params)
    freq_list = token_frequency_list(corpus2) # lista que contem todos os tokens e sua frequencia
    upper_terms_to_remove = int(len(freq_list) * float(upper_cut_percentage))  # qtd. de termos a remover dos mais freq
    lower_terms_to_remove = int(len(freq_list) * float(lower_cut_percentage))  # qtd. de termos a remover dos menos freq
    blacklist = black_list(freq_list, upper_terms_to_remove, lower_terms_to_remove)
    #blacklist.append('aaa')
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


# metodo que le um arquivo de texto e retorna um dicionario com seus textos como par de chave e valor
# retirado de: Uma análise comparativa das ferramentas de pré-processamento de dados textuais: NLTK, PreTexT e R [PPgSI-001/2018]
def loadCorpus(sourcepath):
    corpus = {}
    with open(sourcepath, 'r') as f:
        for line in f:
            if line.strip(): # se a line nao for vazia, line.strip() == true
                corpus[line] = line
    return corpus


# aplica metodos sobre o corpus metodos que o convertem para lower case, tokenizam, removem stopwords e aplicam stemming
# adaptado de: PPgSI-001/2018
def processCorpus(corpus, params):
    (param_foldCase, param_listOfStopWords, param_stemmer) = params
    newCorpus = {}
    for text in corpus: # cada linha em corpus representa um texto, itera pelas linhas
        text = text.rstrip('\n')
        text = foldCase(text, param_foldCase)
        listOfTokens = tokenize(text)
        listOfTokens = removeStopWords(listOfTokens, param_listOfStopWords)
        if (param_stemmer):  # o stemmer nem sempre e utilizado
            listOfTokens = applyStemming(listOfTokens, param_stemmer)

        newCorpus[text] = listOfTokens
    return newCorpus


# converte os textos para lower case
# retirado de: PPgSI-001/2018
def foldCase(sentence, parameter):
    if(parameter): sentence = sentence.lower()
    return sentence


# metodo que divide uma linha em um conjunto de tokens
# retirado de: PPgSI-001/2018
def tokenize(sentence):
    sentence = sentence.replace("_"," ")
    regExpr = '\W+'
    return filter(None, re.split(regExpr, sentence))


# retira os tokens de um texto que nao estao contidos na lista de stopwords
# retirado de: PPgSI-001/2018
def removeStopWords(listOfTokens, listOfStopWords):
    return [token for token in listOfTokens if token not in listOfStopWords]


# utiliza o snowball stemmer do nltk para aplicar o stemming nos tokens de um texto
# retirado de: PPgSI-001/2018
def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]


# recebe um corpus e uma lista de tokens removidos e retorna um mapa onde cada linha contem um texto
# que por sua tambem e um mapa, que mapeia os tokens do texto com seu valor tf-idf se ele for maior que 0
# sendo entao uma representacao esparsa
# adaptado de: PPgSI-001/2018
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
        # exceto os termos contidos na blacklist
        dictOfTfScoredTokens = tf(corpus[document], blacklist)
        # computa um dicionario com o tf-idf score de cada par termo-documento
        corpus[document] = ({token: dictOfTfScoredTokens[token] * idfDict[token] for token in dictOfTfScoredTokens})
    return corpus


# recebe um corpus e uma lista de tokens removidos e retorna um mapa onde cada linha contem um texto
# que por sua tambem e um mapa, que mapeia os tokens do texto com seu valor binario se ele for maior que 0
def representCorpus_binary(corpus, blacklist):
    newCorpus = {}
    for document in corpus:
        newCorpus[document] = binary(corpus[document], blacklist)
    return newCorpus


# recebe um corpus e uma lista de tokens removidos e retorna um mapa onde cada linha contem um texto
# que por sua tambem e um mapa, que mapeia os tokens do texto com seu valor tf se ele for maior que 0
def representCorpus_tf(corpus, blacklist    ):
    newCorpus = {}
    for document in corpus:
        newCorpus[document] = tf(corpus[document], blacklist)
    return newCorpus


# conta os tokens de um texto e o mapeiam com sua frequencia
# adaptado de: PPgSI-001/2018
def tf(listOfTokens, blacklist):
    # cria um dicionario associando cada token com o numero de vezes
    # em que ele ocorre no documento (cujo conteudo eh listOfTokens)
    types = {}
    for token in listOfTokens:
        try:
            types[token] += 1
        except KeyError:
            types[token] = 1

    types = removeTokens(types, blacklist)  # remove tokens cortados
    return types


# conta os tokens de um texto e o mapeiam apenas binariamente
def binary(listOfTokens, blacklist):
    types = {}
    for token in listOfTokens:
        types[token] = 1

    types = removeTokens(types, blacklist)  # remove tokens cortados
    return types


# dada um mapa de tokens e valores, remove os tokens contidos na blacklist
def removeTokens(tokenDict, blacklist):
    for token in blacklist:
        try:
            del tokenDict[token]
        except KeyError:  # o texto nao contem aquele token
            continue
    return tokenDict


# metodo que retorna um mapa com os tokens e sua frequencia no corpus
def token_frequency_list(corpus):
    frequency_list = {}  # dicionario que armazena os tokens
    for text in corpus:  # itera pelos textos do corpus
        # adiciona ao total a contagem tf do texto com uma blacklist vazia
        frequency_list = add_dicts(tf(corpus[text], []), frequency_list)
    return frequency_list


# cria uma lista de que contem os tokens que aparecem um vez no corpus,
# os upper_terms_to_remove-esimos mais frequentes e os lower_terms_to_remove-esimos menos frequentes
def black_list(freq_list, upper_terms_to_remove, lower_terms_to_remove):
    blacklist = []
    for word in freq_list.keys(): # para cada palavra que aparece 1 vez
        if freq_list[word] == 1:
            blacklist.append(word) # adiciona o termo a blacklist

    for word in blacklist:
        del freq_list[word]  # remove o termo da lista para nao ser removido novamente abaixo

    # ordena crescentemente a lista de freq. em uma lista de palavras
    sorted_key_list = sorted(freq_list.items(), key=operator.itemgetter(1))  # retirado de: https://stackoverflow.com/a/613218
    for i in range(lower_terms_to_remove):
        blacklist.append(sorted_key_list[i][0])

    # itera reversamente pela lista, para remover os mais frequentes
    sorted_key_list = list(reversed(sorted_key_list))
    for i in range(upper_terms_to_remove):
        blacklist.append(sorted_key_list[i][0])
    return blacklist


# metodo que adiciona os valores de um dicionario parcial a um dicionario total
def add_dicts(partial_dict, total_dict):
    for token in partial_dict:
        try:
            total_dict[token] += 1
        except KeyError:
            total_dict[token] = 1
    return total_dict


# metodo que representa densamente, em forma de matriz o corpus
# utilizado no som, pois a biblioteca utilizada nao recebe uma representacao esparsa
def represent_matrix(corpus):
    allTokens = []  # cria uma lista com todos os tokens do corpus
    for document in corpus:  # itera pelos textos
        # adiciona os tokens do texto ao total, converte o resultado para set para remover os tokens repetidos
        # e converte o set para lista novamente
        allTokens = list(set(allTokens + corpus[document].keys()))

    new_corpus = []
    new_corpus.append(allTokens)  # adiciona a lista de tokens a primeira linha da matriz
    for text in corpus:
        document = []  # lista que representa o texto
        for token in allTokens:  # adiciona a lista valor de todos os tokens no texto
            try:
                document.append(corpus[text][token])  # adiciona a lista o valor do token na representacao esparsa
            except KeyError:  # o token nao esta contido na representacao esparsa
                document.append(0)  # adiciona o valor 0
        new_corpus.append(document)

    return new_corpus


# armazena o corpus gerado em um arquivo .pkl
# adaptado de: PPgSI-001/2018
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
