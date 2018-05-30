# -*- coding: utf-8 -*-

import sys, random
import cPickle as Pickle
from math import sqrt
import numpy as np
import time

# kmeans
# usage: python means.py [in_filename] [k_int] [epochs_int] [dist_char] [++_binary] [batch_binary] [out_filename]
# example: python means.py corpora/iris.pkl 3 12 s 1 1 out/iris_k3_s_batch

# dist_char:
# d: euclidean distance
# s: cosine similarity

# ++_binary:
# 1: kmeans++ initialization
# 0: random Forgy initialization

# batch_binary:
# 1: batch kmeans
# 0: regular kmeans


# representa um cluster, com um centroide e uma lista de textos
class Cluster:
    # inicializa o Cluster com um centroide que o representa e uma lista de textos
    def __init__(self, centroid):
        self.texts = []
        self.centroid = centroid

    # adiciona um texto a lista de textos que pertencem ao cluster
    def append(self, text):
        self.texts.append(text)

    # limpa a lista de textos que pertencem ao cluster
    def clear(self):
        del self.texts[:]

    # reposiciona o centroid no espaco vetorial no centro de massa de seus textos
    # utilizado no kmeans em batch e xmeans
    def re_center_centroid(self):
        num_of_texts = len(self.texts)
        if num_of_texts < 1:  # caso o cluster nao possui textos
            return  # nao faca nada

        for token in self.centroid:  # itera pelos token do centroide
            self.centroid[token] = 0  # zera o token para utiliza-lo como somador da media

        for text in self.texts:  # itera por cada texto no cluster
            for token in text:  # para cada token no texto
                try:
                    self.centroid[token] += text[token]  # soma o valor do token no texto ao valor final
                except KeyError:  # se o centroide ainda nao possui aquele token
                    self.centroid[token] = text[token]  # inicializa o token com o valor do texto

        for token in self.centroid:  # para cada token resultante
            self.centroid[token] = float(self.centroid[token] / num_of_texts)  # setta o valor para a soma / num_textos

    # calcula a distancia media de um texto para todos os textos em um cluster
    # utilizado no silhouette index
    def avg_dist_to_cluster(self, text):
        dist_sum = 0  # armazena a soma de distancias
        for cluster_text in self.texts:  # itera pelos textos do cluster
            dist_sum += dist(text, cluster_text)  # soma a distancia
        return dist_sum / float((len(self.texts)))  # retorna a distancia media


# kmeans comum
# itera em cada epoca encontrando o bmu de cada texto e imediatamente move o centroide
def kmeans(corpus, k, epochs, verbose=False):
    clusters = generate_clusters(corpus, k)  # inicializa o vetor de k clusters
    # cria lista que armazena o antigo cluster de cada texto, para parar caso nao haja mudanca nos clusters
    text_previous_bmu_list = create_text_previous_bmu_list(corpus)
    for n in (range(epochs)):  # loop de epocas
        changes = 0  # quantidade de mudancas nos clusters
        clear_clusters(clusters)  # limpa a lista de textos contidos em cada cluster
        for text in corpus.keys():  # itera sobre os textos do corpos
            bmu = find_bmu(clusters, corpus[text])  # encontra o BMU
            bmu.append(corpus[text])  # insere o texto no cluster, para obter metricas e fazer o silhouette
            move_centroid(corpus[text], bmu)  # move o centroide para a media entre o texto e o centroiude
            if bmu != text_previous_bmu_list[text]:  # se o BMU atual e diferente do antigo
                text_previous_bmu_list[text] = bmu  # atualiza a lista de antigo BMU
                changes += 1  # incrementa o numero de mudancas
        if verbose:  # imprime informacao sobre o tamanho dos clusters, se verbose == True
            print("epoch ", n)
            for cluster in clusters:
                print(len(cluster.texts))
        if changes < 1:  # se nao houve mudancas natodo: comment epoca, para o aprendizado
            break
    return clusters  # retorna o vetor de clusters


# kmeans em batch
# itera em cada epoca encontrando o bmu de cada texto e move o centroide no final da epoca
def batch_kmeans(corpus, k, epochs, verbose=False):
    clusters = generate_clusters(corpus, k)  # inicializa o vetor de k clusters
    # cria lista que armazena o antigo cluster de cada texto, para parar caso nao haja mudanca nos clusters
    text_previous_bmu_list = create_text_previous_bmu_list(corpus)
    for n in (range(epochs)):  # loop de epocas
        changes = 0    # quantidade de mudancas nos clusters
        clear_clusters(clusters)  # limpa a lista de textos contidos em cada cluster
        for text in corpus.keys():  # itera sobre os textos do corpos
            bmu = find_bmu(clusters, corpus[text])  # encontra o BMU
            bmu.append(corpus[text])   # insere o texto no cluster, para obter metricas, silhouette e mover o centroide
            if bmu != text_previous_bmu_list[text]:  # se o BMU atual e diferente do antigo
                text_previous_bmu_list[text] = bmu  # atualiza a lista de antigo BMU
                changes += 1  # incrementa o numero de mudancas
        re_center_centroids(clusters)  # move o centroide para o centro de massa do seu cluster
        if verbose:  # imprime informacao sobre o tamanho dos clusters, se verbose == True
            print("epoch ", n)
            for cluster in clusters:
                print(len(cluster.texts))
        if changes < 1:  # se nao houve mudancas na epoca, para o aprendizado
            break
    return clusters  # retorna o vetor de clusters


# metodo do kmeans convencional que posiciona o centroide na media entre o centroide e o texto
def move_centroid(text, bmu):
    tokens = set(text) | set(bmu.centroid)  # a uniao dos tokens de text e bmu.centroid
    for token in tokens:  # itera sobre a uniao
        # A.get(token, 0) retornara o valor de token em a, se A possuir algum valor para token, ou 0
        bmu.centroid[token] = (bmu.centroid.get(token, 0) + text.get(token, 0)) / 2  # armazena a media em centroid[token]


# retorna o cluster cujo centroide esta mais proximo de text
def find_bmu(clusters, text):
    min_distance = sys.maxsize  # inicializa a distancia como o maior int possivel
    closest_cluster = None  # inicializa o centroid mais prox como nulo
    for cluster in clusters:  # itera pelos clusters
        current_dist = dist(cluster.centroid, text)  # calcula a distancia entre o centroide e o texto
        if current_dist < min_distance:  # atualiza os minimos se a distancia encontrada for a menor ate agora
            min_distance = current_dist
            closest_cluster = cluster
    return closest_cluster  # retorna c cluster vencedor


# retorna a distancia euclideana entre um texto e outro texto
# adaptado de: https://stackoverflow.com/a/38370159
def dist_euclid(text1, text2):
    # para cada token na uniao entre tokens to text1 e text2, soma a diferenca ao quadrado e retorna a raiz
    return sqrt(sum((text1.get(token, 0) - text2.get(token, 0)) ** 2 for token in set(text1) | set(text2)))


# retorna a distancia de simil   aridade de cosseno entre um texto e outro
# adaptado de: https://stackoverflow.com/a/38370159
def dist_cos_simil(text1, text2):
    # para cada token na uniao entre tokens to text1 e text2, soma a multiplicacao dos valores
    t_sum = sum([text1.get(token, 0)*text2.get(token, 0) for token in set(text1) | set(text2)])
    # retorna 1 menos a soma dividida pela magnitude multiplicada de cada texto, que foi pre calculada
    return 1 - (t_sum / (magnitude(text1) * magnitude(text2)))


# metodo que pre calcula a magnitude dos vetores no corpus para o calculo de similaridade de cosseno
def magnitude(text):
    # para cada token no documento, soma o quadrado do valor e retorna a raiz da soma
    magnitude = sqrt(sum([text[token] ** 2 for token in text]))
    if magnitude > 0: return magnitude
    return 1


# metodo que inicializa aleatoriamente k clusters
def random_init(corpus, k):
    centroids = random_centroids(corpus, k)  # recebe os centroides aleatorios
    clusters = []  # cria um vetor de clusters
    for centroid in centroids:  # itera sobre os centroides
        clusters.append(Cluster(centroid))  # cria clusters com os centroides
    return clusters  # retorna os clusters


# aloca aleatoriamente k centroides,
# seguindo o metodo de Forgy, onde um centroide assume os mesmos valores de um dado aleatorio
def random_centroids(corpus, k):
    centroids = []  # cria um vetor de centroides
    for random_text in random.sample(corpus, k):  # itera sobre um vetor de k textos aleatorios
        centroid = corpus[random_text].copy()  # cria um centroide que copia o texto
        centroids.append(centroid)  # adiciona ao vetor de centroides
    return centroids  # retorna os centroides


# metodo que inicializa k clusters seguindo o metodo kmeans++
def plus_plus_init(corpus, k):
    clusters = []  # cria um vetor de clusters
    corpus = corpus.copy()  # cria uma copia do corpus para remover textos sem afeta-lo
    first_centroid = random.choice(corpus.keys())  # escolhe um texto inicial aleatorio
    clusters.append(Cluster(corpus[first_centroid].copy()))  # cria um cluster inicial copiando o texto
    del corpus[first_centroid] # impede que o mesmo texto seja escolhido novamente, deletando-o do corpus
    for i in range(k - 1):  # gera mais k-1 clusters
        item_prob = []  # cria um vetor que armazena a probabilidade de cada texto no corpus ser escolhido
        total_dist = 0  # armazena a distancia total entre textos e seus centroides mais proximos
        for text in corpus:  # itera sobre os textos do corpus restante
            closest_centroid = find_bmu(clusters, corpus[text])  # encontra o centroide mais proximo
            closest_cluster_dist = dist(closest_centroid.centroid, corpus[text])  # calcula a distancia a ele
            prob = closest_cluster_dist ** 2  # calcula a distancia ao quadrado, utilizada para calcular a probabilidade
            item_prob.append(prob)  # armazena esta distancia
            total_dist += prob  # soma a distancia ao total
        for j in range(len(item_prob)):  # itera sobre as distancias armazenadas
            # divide as distancias pelo total, de maneira que soma cumulativa das probabilidades de 1
            item_prob[j] = item_prob[j] / total_dist

        # escolhe um texto de acordo com as probabilidade de item_prob.
        # np.random.choice retorna um vetor entao o texto retornado e acessado com o [0]
        new_centroid = np.random.choice(corpus.keys(), 1, p=item_prob)[0]
        clusters.append(Cluster(corpus[new_centroid].copy()))  # adiciona um cluster cujo centroide copia o texto
        del corpus[new_centroid]  # impede que o mesmo texto seja escolhido novamente
    return clusters  # retorna os clusters


# metodo utilizado no kmeans em batch que re centraliza os centroides em seu cluster
def re_center_centroids(clusters):
    for cluster in clusters:  # itera por cada cluster
        cluster.re_center_centroid()  # re centraliza seu centroide


# metodo utilizado nas epocas dos kmeans que limpa a lista de textos que compoem um cluster
def clear_clusters(clusters):
    for cluster in clusters:  # itera por cada cluster
        cluster.clear()  # limpa a lista


# calcula o indice de silhouette de uma determinada estrutura
def silhouette(clusters, verbose=False):
    silhouette_sum = 0  # soma que armazena o silhouette de todos os textos
    total_texts = 0  # total de textos
    for cluster in clusters:  # itera pelos clusters
        num_texts = len(cluster.texts)
        total_texts += num_texts
        if num_texts < 1:  # se nao ha textos no cluster
            if verbose: print('cluster ', cluster, ' is empty')
            silhouette_sum += -1  # soma -1 na soma para desencorajar clusters vazios durante o xmeans
            continue
        other_clusters = clusters[:]  # copia o array de clusters para remover items sem destrui-lo
        other_clusters.remove(cluster)  # remove o cluster atual
        # encontra apenas o centroide mais proximo do centroide do cluster atual e toma isto como o cluster mais proximo
        # ao inves de calcular a distancia media para cada outro cluster para economizar tempo e obter resultados aprox.
        closest_cluster = find_bmu(other_clusters, cluster.centroid)
        while len(closest_cluster.texts) < 1:  # se o cluster mais proximo esta vazio
            other_clusters.remove(closest_cluster)  # remove o cluster vazio da lista
            closest_cluster = find_bmu(other_clusters, cluster.centroid) # procura o segundo mais proimo
        cluster_silhouette_sum = 0  # variavel que armazena o silhouette do cluster para impressao
        for text in cluster.texts:  # para cada texto no cluster
            cluster_avg_dist = cluster.avg_dist_to_cluster(text)  # encontra o valor de a(texto)
            closest_cluster_avg_dist = closest_cluster.avg_dist_to_cluster(text)  # encontra o valor de b(texto)
            text_silhouette = ((closest_cluster_avg_dist - cluster_avg_dist)  # calcula b-a/max(a,b)
                                       / max(closest_cluster_avg_dist, cluster_avg_dist))
            cluster_silhouette_sum += text_silhouette  # adiciona ao silhouette do cluster
            silhouette_sum += text_silhouette  # adiciona ao silhouette total
        cluster_silhouette = cluster_silhouette_sum / num_texts  # calcula o silhouette medio do cluster
        if verbose:  # imprime se necessario
            print('silhouette for ', cluster, len(cluster.texts), cluster_silhouette)
    return silhouette_sum / total_texts  # retorna o silhouette medio dos clusters


# metodo que cria uma lista que contem o antigo cluster ao qual um texto pertencia
# utilizado para determinar a quantidades de mudancas na estrutura em uma epoca do kmeans
def create_text_previous_bmu_list(corpus):
    list = {}  # inicializa a lista
    for document in corpus.keys():  # para cada texto
        list[document] = None  # inicializa o valor como None
    return list  # retorna o vetor


# retorna todos os tokens unicos em um corpus
def list_of_all_tokens(corpus):
    all_tokens = []  # lista que armazena todos os tokensUnicodeDecodeError: 'ascii' codec can't decode byte
    for document in corpus.values():  # itera sobre todos os documentos
        all_tokens = list(set(all_tokens + document.keys()))  # adiciona todos os tokens de documento a lista
    return list(set(all_tokens))  # converte a lista para set para eliminar repeticoes e converte para lista de novo


# le o .pkl em file_name e retorna o objeto carregado
# utilizado para ler o corpus gerado por pre.py
def unpickle(file_name):
    f = open(file_name, 'r')
    p = Pickle.Unpickler(f)
    return p.load()


# metodo que armazena os clusters gerados em 'name'.pkl
def serialise(obj, name):
    f = open(name + '.pkl', 'wb')
    p = Pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()


# main
if __name__ == '__main__':
    start = time.clock()  # armazena o tempo atual

    read_file = sys.argv[1]  # nome do arquivo de entrada
    corpus = unpickle(read_file)  # le o arquivo

    k = int(sys.argv[2])  # numero de clusters a gerar
    epochs = int(sys.argv[3])  # numero maximo de epocas
    write_file = sys.argv[7]  # nome do arquivo de saida

    if sys.argv[4] == 's':  # se a distancia a ser utilizada for similaridade de cossenos
        dist = dist_cos_simil
    else:
        dist = dist_euclid  # else a distancia euclideana

    if sys.argv[5] == 1:  # se a inicializacao for com o kmeans++
        generate_clusters = plus_plus_init
    else:
        generate_clusters = random_init  # ou aleatoria

    if sys.argv[6] == 1:  # se o kmeans for em batch
        clusters = batch_kmeans(corpus, k, epochs, True)
    else:
        clusters = kmeans(corpus, k, epochs, True)  # ou convencional

    serialise(clusters, write_file)  # armazena os clusters em um .pkl
    print('total silhouette for ', sys.argv, ': ', silhouette(clusters, True), 'k: ', len(clusters))  # calcula o silhouette

    end = time.clock()  # armazena o tempo atual
    print('tempo: ', end-start)  # imprime o delta tempo
