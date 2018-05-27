# -*- coding: utf-8 -*-

import sys, random
import cPickle as Pickle
from math import sqrt
import time

# usage: python means.py [in_filename] [k_int] [epochs_int] [dist_char] [out_filename]
# example: python means.py reuters_tf_idf.pkl 3 25 s out
# d: euclidean distance
# s: cosine similarity


def main():
    # todo: comment
    read_file = sys.argv[1]
    k = int(sys.argv[2])
    epochs = int(sys.argv[3])
    write_file = sys.argv[5]
    corpus = unpickle(read_file)
    if (sys.argv[4] == 's'):
        dist = dist_cos_simil
        magnitude = set_magnitude(corpus)
    clusters = batch_kmeans(corpus, k, epochs)
    print('total silhouette: ', silhouette(clusters))
    serialise(clusters, write_file)


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
    def re_center_centroid(self):
        num_of_texts = len(self.texts)
        if num_of_texts < 1:  # caso o cluster nao possui textos
            return  # nao faca nada
        for token in self.centroid:  # itera por cada token no centroide
            new_token_value_sum = 0  # guarda a soma dos valores que o token assume nos textos
            for text in self.texts: # itera pelos textos do cluster
                try:
                    new_token_value_sum += float(text[token]) # adiciona a sum o valor do token no texto
                except KeyError:  # o texto nao contem aquele token
                    new_token_value_sum += 0
            self.centroid[token] = float(new_token_value_sum / num_of_texts)  # setta para o valor medio no cluster

    # calcula a distancia media de um texto para todos os textos em um cluster
    # utilizado no silhouette index
    def avg_dist_to_cluster(self, text):
        dist_sum = 0 # armazena a soma de distancias
        for cluster_text in self.texts: # itera pelos textos do cluster
            dist_sum += dist(text, cluster_text) # soma a distancia
        return dist_sum / float((len(self.texts))) # retorna a distancia media


# todo: comment
def unpickle(file_name):
    f = open(file_name, 'r')
    p = Pickle.Unpickler(f)
    return p.load()


# retorna todos os tokens unicos em um corpus
def list_of_all_tokens(corpus):
    all_tokens = [] # lista que armazena todos os tokens
    for document in corpus.values():  # itera sobre todos os documentos
        all_tokens = list(set(all_tokens + document.keys()))  # adiciona todos os tokens de documento a lista
    return list(set(all_tokens))  # converte a lista para set para eliminar repeticoes e converte para lista de novo


# todo: commenttext_dist_simil
def batch_kmeans(corpus, k, epochs):
    clusters = generate_clusters(corpus, k)
    for n in (range(epochs)):  # loop de epocas
        clear_clusters(clusters)
        for text in corpus.values():
            bmu = find_bmu(clusters, text)
            bmu.append(text)
        re_center_centroids(clusters)
        print("epoch ", n)
        for cluster in clusters:
        #    print(len(cluster.texts))
            if len(cluster.texts) < 1: sys.exit(-1)
    return clusters


# retorna o cluster cujo centroide esta mais proximo de text
def find_bmu(clusters, text):
    min_distance = sys.maxsize # inicializa a distancia como o maior int possivel
    closest_cluster = None # inicializa o centroid mais prox como nulo
    for cluster in clusters: # itera pelos clusters
        current_dist = dist(cluster.centroid, text) # calcula a distancia entre o centroide e o texto
        if current_dist < min_distance: # atualiza os minimos se a distancia encontrada for a menor ate agora
            min_distance = current_dist
            closest_cluster = cluster
    return closest_cluster # retorna c cluster vencedor


# todo: comment
# retorna a distancia euclideana entre um texto e outro texto
# https://stackoverflow.com/a/38370159
def dist_euclid(text1, text2):
    return sqrt(sum((text1.get(token, 0) - text2.get(token, 0)) ** 2 for token in set(text1) | set(text2)))


# todo: comment
# https://stackoverflow.com/a/38370159
def dist_cos_simil(text1, text2):
    sum = sum(text1.get(token, 0)*text2.get(token, 0) for token in set(text1) | set(text2))
    return 1 - (sum / (magnitude[text1] * magnitude[text2]))


# todo: comment
def generate_clusters(corpus, k):
    all_tokens = list_of_all_tokens(corpus)
    centroids = random_init(all_tokens, k)
    clusters = []
    for centroid in centroids:
        clusters.append(Cluster(centroid))
    return clusters


# todo: comment
def random_init(all_tokens, k):
    centroids = []
    random.seed()
    for i in range(k):
        centroid = {}
        for token in all_tokens:
            centroid[token] = random.uniform(0, 1)
        centroids.append(centroid)
    return centroids


# todo: comment
def re_center_centroids(clusters):
    for cluster in clusters:
        cluster.re_center_centroid()


# todo: comment
def clear_clusters(clusters):
    for cluster in clusters:
        cluster.clear()


# todo: comment
def silhouette(clusters):
    silhouette_sum = 0
    for cluster in clusters:
        if len(cluster.texts) < 1:
            print('cluster ', cluster, ' is empty')
            continue
        other_clusters = clusters[:]  # shallow copy of clusters array
        other_clusters.remove(cluster)
        closest_cluster = find_bmu(other_clusters, cluster.centroid)
        while len(closest_cluster.texts) < 1:
            other_clusters.remove(closest_cluster)
            closest_cluster = find_bmu(other_clusters, cluster.centroid)
        cluster_silhouette_sum = 0
        for text in cluster.texts:
            cluster_avg_dist = cluster.avg_dist_to_cluster(text)  # a(text)
            closest_cluster_avg_dist = closest_cluster.avg_dist_to_cluster(text)  # b(text)
            cluster_silhouette_sum += ((closest_cluster_avg_dist - cluster_avg_dist)
                                       / max(closest_cluster_avg_dist, cluster_avg_dist))

        cluster_silhouette = cluster_silhouette_sum / len(cluster.texts)
        print('silhouette for ', cluster, ', ', len(cluster.texts), cluster_silhouette)
        silhouette_sum += cluster_silhouette
    return silhouette_sum / len(clusters)


# todo: comment
def set_magnitude(corpus):
    for document in corpus:
        magnitude[document] = sqrt(sum([corpus[document][token] ** 2 for token in corpus[document]]))
    return set_magnitude


def serialise(obj, name):
    f = open(name + '.pkl', 'wb')
    p = Pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()

if __name__ == '__main__':
    start = time.clock()
    dist = dist_euclid # setta a distancia euclideana como padrao
    magnitude = {} # declara um dicionario que armazenara a magnitude dos vetores para a similaridade de cossenos
    main()
    end = time.clock()
    print(end-start)
