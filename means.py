# -*- coding: utf-8 -*-

import sys, random
import cPickle as Pickle
from math import sqrt

# usage: python means.py [in_filename] [k_int] [epochs_int] [out_filename]
# example: python means.py reuters_tf_idf.pkl 3 25 out

def main():
    read_file = sys.argv[1]
    k = int(sys.argv[2])
    epochs = int(sys.argv[3])
    #write_file = sys.argv[4]
    corpus = unpickle(read_file)
    clusters = batch_kmeans(corpus, k, epochs)
    print('total silhouette: ', silhouette(clusters))


# representa um cluster, com um centroide e uma lista de textos
class Cluster:
    def __init__(self, centroid):
        self.texts = []
        self.centroid = centroid

    def append(self, text):
        self.texts.append(text)

    def clear(self):
        del self.texts[:]

    def re_center_centroid(self):
        num_of_texts = len(self.texts)
        if num_of_texts < 1:
            return
        for token in self.centroid:
            new_token_value_sum = float(self.centroid[token])
            for text in self.texts:
                try:
                    new_token_value_sum += float(text[token])
                except KeyError:  # o texto nao contem aquele token
                    new_token_value_sum += 0
            self.centroid[token] = float(new_token_value_sum / num_of_texts)

    def avg_dist_to_cluster(self, text):
        avg_dist = 0
        for txt in self.texts:
            avg_dist += dist(text, txt)
        return avg_dist / float((len(self.texts)))


def unpickle(file_name):
    f = open(file_name, 'r')
    p = Pickle.Unpickler(f)
    return p.load()


# retorna todos os tokens unicos em um corpus
def list_of_all_tokens(corpus):
    all_tokens = []
    for document in corpus:  # itera sobre todos os documentos
        all_tokens = list(set(all_tokens + corpus[document].keys()))  # adiciona todos os tokens de documento a lista
    return list(set(all_tokens))  # retorna todos os tokens sem repeticao


def batch_kmeans(corpus, k, epochs):
    clusters = generate_clusters(corpus, k)
    for n in (range(epochs)):  # loop de epocas
        clear_clusters(clusters)
        for text in corpus:
            bmu = find_bmu(clusters, corpus[text])
            bmu.append(corpus[text])
        re_center_centroids(clusters)
        #print("epoch ", n)
        #for cluster in clusters:
        #    print(len(cluster.texts))
        #    if len(cluster.texts) < 1: sys.exit(-1)
    return clusters


def find_bmu(clusters, text):
    min_distance = sys.maxsize
    closest_centroid = None
    for cluster in clusters:
        current_dist = dist(cluster.centroid, text)
        if current_dist < min_distance:
            min_distance = current_dist
            closest_centroid = cluster
    return closest_centroid


def dist(text1, text2):
    sum = 0
    for token in text1:
        try:
            sum += (float(text1[token]) - float(text2[token])) ** 2
        except KeyError:  # o texto2 nao contem aquele token
            sum += float(text1[token] - 0) ** 2
    return sqrt(sum)


def generate_clusters(corpus, k):
    all_tokens = list_of_all_tokens(corpus)
    centroids = random_init(all_tokens, k)
    clusters = []
    for centroid in centroids:
        clusters.append(Cluster(centroid))
    return clusters


def random_init(all_tokens, k):
    centroids = []
    random.seed()
    for i in range(k):
        centroid = {}
        for token in all_tokens:
            centroid[token] = random.uniform(0, 1)
        centroids.append(centroid)
    return centroids


def re_center_centroids(clusters):
    for cluster in clusters:
        cluster.re_center_centroid()


def clear_clusters(clusters):
    for cluster in clusters:
        cluster.clear()

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
        print('silhouette for ', cluster, ': ', cluster_silhouette)
        silhouette_sum += cluster_silhouette
    return silhouette_sum / len(clusters)


if __name__ == '__main__':
    main()
