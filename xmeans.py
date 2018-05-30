# -*- coding: utf-8 -*-

import sys
import means
from means import Cluster

# xmeans
# usage: python xmeans.py [in_filename] [corpus_filename] [k_max] [dist_char] [out_filename]
# example: python xmeans.py out/iris_k3_s_batch.pkl corpora/iris.pkl 7 s out/iris_x

# dist_char:
# d: euclidean distance
# s: cosine similarity

# recebe um vetor de clusters e um k_max e divide os clusters enquanto a nova configuracao
# apresentar melhora no silhouette index da classificacao ou obter k_max clusters
def xmeans(clusters, k_max):
    result_clusters = clusters  # cria um vetor que armazena os clusters resultantes
    splittable_clusters = clusters  # cria um vetor que armazena os clusters que ainda podem ser divididos
    # loop de Improve Structure, divide cada cluster em dois e avalia a estrutura resultante
    while len(result_clusters) < k_max:  # enquanto o numero de clusters e menor que k_max
        new_clusters = []  # vetor que armazena novos clusters gerados para atribuir a splittable_clusters no prox. loop
        for cluster in splittable_clusters:  # itera sobre os clusters que podem dividir-se
            # tenta dividir o cluster, enviando o cluster a dividir e uma copia da estrutura atual
            # o metodo split_cluster retornara 2 novos clusters se a divisao melhorar a estrutura
            # ou nao retornara nada
            splits = split_cluster(cluster, result_clusters[:])
            if splits:  # se o metodo split_cluster retornou novos clusters, a divisao melhorou a estrutura
                new_clusters = new_clusters + splits  # adicionaremos os novos clusters aos splittable_clusters
                result_clusters = result_clusters + splits  # adiciona os novos clusters ao resultado
                result_clusters.remove(cluster)  # remove o cluster dividido da estrutura
        splittable_clusters = new_clusters  # atualiza splittable_clusters para dividir os novos clusters
        if not splittable_clusters:  # se nao ha clusters divisiveis
            return result_clusters  # retorna a estrutura atual
    return result_clusters  # retorna a estrutura encontrada


# recebe um cluster a ser dividido e uma lista de clusters que representam a atual estrutura
# divide o cluster e avalia a estrutura resultante com o indice silhouette
# retorna os clusters filhos se houve melhora no indice ou uma lista vazia se nao houve
def split_cluster(cluster, clusters):
    old_silhouette = means.silhouette(clusters)  # calcula o silhouette da estrutura atual
    new_clusters = place_new_centroids(cluster)  # cria dois novos clusters filhos
    for text in cluster.texts:  # itera sobre os textos do cluster pai
        bmu = means.find_bmu(new_clusters, text)  # encontra o centroide mais proximo, entre os dois filhos
        bmu.append(text)  # adiciona o texto ao cluster do BMU
    clusters.remove(cluster)  # remove o cluster pai da estrutura
    clusters = clusters + new_clusters  # adiciona os dois filhos
    for n_cluster in new_clusters:  # itera sobre os filhos
        n_cluster.re_center_centroid()  # centraliza os centroides no cluster
    new_silhouette = means.silhouette(clusters)  # calcula o novo indice silhouette
    if new_silhouette > old_silhouette:  # se houve melhora
        return new_clusters  # retorna os filhos
    return []  # se nao houve, retorna uma lista vazia


# metodo que divide um cluster em dois filhos gerando dois centroides
# encontrando, para cada token a maior distancia entre o valor do token no centroide e o valor do token em um texto
# um dos centroides recebera o valor atual do centroide mais a maior distancia
# o outro recebera o valor atual do centroide menos a maior distancia
# posicionando algum dos centroides na borda mais distante do cluster na dimensao deste token
# e o outro na direcao oposta
def place_new_centroids(cluster):
    positive_centroid = {}  # inicializa os novos centroides
    negative_centroid = {}  # inicializa os novos centroides
    for token in cluster.centroid.keys():  # itera sobre os tokens do centroide
        max_dist = 0  # variavel que armazena a maior distancia encontrada no token
        for text in cluster.texts:  # itera pelos textos do cluster
            try:
                # calcula a diferenca entre o valor do token no centroide e no texto
                dist = abs(text[token] - cluster.centroid[token])
                max_dist = dist if dist > max_dist else max_dist  # atualiza o maximo se necessario
            except KeyError:  # caso o texto nao possua este token
                continue  # nao faca nada
        positive_centroid[token] = cluster.centroid[token] + max_dist  # incrementa a distancia no valor do token
        negative_centroid[token] = max(cluster.centroid[token] - max_dist, 0)  # decrementa a distancia. pode dar < 0
    return [Cluster(positive_centroid), Cluster(negative_centroid)]  # retorna os novos clusters


# main
if __name__ == '__main__':
    read_file = sys.argv[1]  # nome do arquivo de clusters
    corpus_file = sys.argv[2]  # nome do arquivo do corpus
    clusters = means.unpickle(read_file)  # le os clusters
    means.corpus = means.unpickle(corpus_file)  # le o corpus
    k_max = sys.argv[3]  # numero maximo de clusters a gerar
    if sys.argv[4] == 's':  # determina a distancia a ser utilizada
        means.dist = means.dist_cos_simil
    else:
        means.dist = means.dist_euclid
    write_file = sys.argv[5]  # nome do arquivo de saida
    clusters = xmeans(clusters, k_max)  # executa o xmeans
    print means.silhouette(clusters, True)