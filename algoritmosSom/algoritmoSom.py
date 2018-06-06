# coding: utf-8
## Implementação do Algoritmo SOM

import os, sys
import math
import matplotlib.pylab as plt
import pickle as pickle
import pandas as pd
import numpy as np
import operator

from minisom import MiniSom

# classe de modelagem e treinamento
class Som:
    def __init__(self, data, filename):
        self.erroQuantizacao = {}
        self.data = data
        self.filename = filename
        
        # definicao dos parametros
        # dimensao da rede
        self.dimensao = list()
        mapSize = 5*int(math.sqrt(len(data)))
        self.dimensao.append(int(math.sqrt(mapSize/4))) #pequeno
        self.dimensao.append(int(math.sqrt(mapSize)))   #medio
        self.dimensao.append(int(math.sqrt(mapSize*2))) #grande
        
        # taxa de aprendizado
        self.aprendizado = [0.2, 0.5, 0.8]
        
        # distancia de vizinhos baseada no tamanho de cada rede
        self.sigma = [0, 1, 2]
        for index, elem in enumerate(self.dimensao):
            self.sigma[index] = list()
            self.sigma[index].append(int(elem/2))
            self.sigma[index].append(int(elem/4))
            self.sigma[index].append(int(elem/8))

        # numero de iteracoes/epoca
        self.epocas = [100, 1000, 2000]
    
    def treinar(self, dim, sigma, step, epochs):
        som = MiniSom(dim, dim, len(self.data[0]), sigma=sigma, learning_rate=step, random_seed=450)
        som.train_batch(self.data, epochs)

        path = 'trained-models/' + self.filename + '(dim_' + str(dim) + '-sigma_' + str(sigma) + '-step_' + str(step) + '-epocas_' + str(epochs) + ')'
        
        #guarda erros de quantizacao por arquivo
        self.erroQuantizacao[path] = som.quantization_error(self.data)
        
        #salva modelo treinado
        np.save(path, som.get_weights())

    # Exibe erros de Quantizacao ordenados
    def exibiErros(self):
        print(sorted(self.erroQuantizacao.items(), key=operator.itemgetter(1)))

    def run(self, test=False):
        if not test:
            for index, dim in enumerate(self.dimensao):
                for sig in self.sigma[index]:
                    for step in self.aprendizado:
                        for epochs in self.epocas:
                            self.treinar(dim, sig, step, epochs)
            self.exibiErros()
        else:
            #caso de teste
            self.treinar(20, 3, 0.2, 1)


## Pipeline de processamento do SOM

# Retorna valores do corpus
def loadData(sourcePath):
    with open('corpora/' + sourcePath, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)

    new_header = df.iloc[0] #pega primera linha para cabecalho
    df = df[1:]             #pega os dados abaixo da primeira linha
    df.columns = new_header #atribui a primeira linha como cabecalho

    df = pd.DataFrame(df.values, columns=df.columns, dtype='float32') #transforma types em inteiros
    return df.values

# Executa algoritmo SOM para arquivos passados
def run(arquivosAProcessar = list(), teste=False):
    for file in arquivosAProcessar:
        data = loadData(file + '.pkl')
        som = Som(data, file)
        som.run(teste)

# Define arquivos a serem processados
def main(test=False):
    arquivosAProcessar = list()
    if not test:
        corpora = ['process', 'reuters']
        types = ['-binary', '-tf', '-tfidf']
        corte = ['-corte']
        stemming = ['-stemmer','']
        for corpus in corpora:
            for file in types:
                for cut in corte:
                    for stemmer in stemming:
                        name = file + stemmer + cut
                        arquivosAProcessar.append(corpus + '/som/' + corpus + name + '_som')
                        
    else:
        #caso de teste
        arquivosAProcessar.append('reuters/som/reuters-binary-corte_som')

    # processa lista de arquivos
    run(arquivosAProcessar, test)

if __name__ == "__main__":
    main(False)

