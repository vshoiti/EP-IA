# coding: utf-8
import random

# carrega dados do corpus original
def loadCorpus(sourcepath):
    corpus = list()
    with open(sourcepath, 'r', encoding="ISO-8859-1") as f:
        for line in f:
            if line.strip(): # se a line nao for vazia, line.strip() == true
                corpus.append(line)
    return corpus

data = loadCorpus('raw/reuters.txt')

# gera amostra randomica
data = random.sample(data, 5500)

# concatena conteudo a ser gravado
toWrite = ''
for text in data:
    text = text.strip()
    toWrite += text + '\n'

# grava amostra em arquivo
with open('raw/reuters-cortado.txt', 'w') as f:
    f.write(toWrite)

