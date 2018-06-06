# coding: utf-8

## Implementação do Algoritmo SOM

import os, sys
import math

import pickle as pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from pathlib import Path
from minisom import MiniSom

## Classe SOM
# classe de modelagem e treinamento
class Som:
    def __init__(self, data, sourceModel):
        self.data = data
        self.sourceModel = sourceModel
        self.som = None
    
    # carrega modelo em uma variavel som
    def criaSom(self, dim, sigma, step):
        file = Path(self.sourceModel)
        
        # verifica se modelo esta na pasta
        if file.is_file():
            self.som = MiniSom(dim, dim, len(self.data[0]), sigma=sigma, learning_rate=step, random_seed=450)

            #carrega pesos treinados no modelo
            pesos = np.load(self.sourceModel)
            self.som._weights = pesos

    # mostra erro de quantizacao
    def calculaErros(self):
        erro = self.som.quantization_error(self.data)
        print(erro)
    
    # Plota Matrix U
    def mapaDistancia(self):
        sns.heatmap(self.som.distance_map()).set_title("Matrix U")
        plt.show()
        
    # Plota Mapa de calor dos neuronios de saída
    def mapaCalor(self):
        sns.heatmap(self.som.activation_response(self.data)).set_title("Mapa de Calor")
        plt.show()

    def run(self, dim, sig, step):
        self.criaSom(dim, sig, step)

        # Mostra erro de Quantização
        self.calculaErros()

        # Exibi graficos
        self.mapaDistancia()
        self.mapaCalor()


## Pipeline de processamento do SOM

# carrega dados do arquivo passado
def loadData(sourcePath):
    with open(sourcePath, 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)

    new_header = df.iloc[0] #pega primera linha para cabecalho
    df = df[1:] #pega os dados abaixo da primeira linha
    df.columns = new_header #atribui a primeira linha como cabecalho

    df = pd.DataFrame(df.values, columns=df.columns, dtype='int16') #transforma types em inteiros
    return df.values

# Define arquivos a serem processados
def main(test=False):
    if test:
        #caso de teste
        sourceData  = 'corpora/process/som/process-binary-stemmer-corte_som.pkl'
        sourceModel = 'trained-models/process/som/process-binary-stemmer-corte_som(dim_19-sigma_2-step_0.2-epocas_2000).npy'
        dimensoes = 19
        sigma = 2
        step = 0.2
    else:
        sourceData = sys.argv[1]
        sourceModel = sys.argv[2]
        dimensoes = sys.argv[3]
        sigma = sys.argv[4]
        step = sys.argv[5]
    
    data = loadData(sourceData)

    # Cria modelo para geracao das metricas necessarias para plotagem
    som = Som(data, sourceModel)
    som.run(dimensoes, sigma, step)

if __name__ == "__main__":
    main(False)
