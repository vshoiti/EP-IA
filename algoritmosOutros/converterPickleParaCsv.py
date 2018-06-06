# coding: utf-8
import pandas as pd
import pickle as pickle

# converte dados em Pickle para csv
def converter(sourcePath):
    #recupera dados em Pickle
    with open('corpora/' + sourcePath  + '.pkl', 'rb') as f:
        data = pickle.load(f)

    # converte para dataFrame
    df = pd.DataFrame(data)

    new_header = df.iloc[0] #pega primera linha para cabecalho
    df = df[1:] #pega os dados abaixo da primeira linha
    df.columns = new_header #atribui a primeira linha como cabecalho

    df = pd.DataFrame(df.values, columns=df.columns, dtype='float64') #transforma types em inteiros
    # salva em csv
    df.to_csv('corporaCsv/' + sourcePath + '.csv', index=False)

# Define arquivos a serem Convertidos
def main():
    arquivosAProcessar = list()
    corpora = ['reuters', 'process']
    types = ['-binary', '-tf', '-tfidf']
    corte = ['-corte','']
    stemming = ['-stemmer','']
    algoritmos = ['som', 'kmeans']
    tipoArquivo = ['_som', '']
    
    #pega todas as versões dos arquivos
    for corpus in corpora:
        for typeAl in algoritmos:
            for file in types:
                for cut in corte:
                    for stemmer in stemming:
                        typeEnd = ''
                        if typeAl == 'som': typeEnd = '_som'
                        name = file + stemmer + cut
                        arquivosAProcessar.append(corpus + '/' + typeAl + '/' + corpus + name + typeEnd)

    # executa funcao converter para todos os arquivos
    for file in arquivosAConverter:
        converter(file)

if __name__ == "__main__":
    main()