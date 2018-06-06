# -*- coding: utf-8 -*-

import means
from means import Cluster
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


# carrega o corpus de file_name.pkl e adiciona uma chave target cujo valor e o cluster do texto
def load_corpus (file_name):
    clusters = means.unpickle(file_name)
    corpus = []
    i = 0
    for cluster in clusters:
        i += 1
        for text in cluster.texts:
            text['TARGET'] = i
            corpus.append(text)
    return corpus

file_name = sys.argv[1]
corpus = load_corpus(file_name)

# adaptado de: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
df = pd.DataFrame(corpus)
df = df.fillna(0).to_sparse(fill_value=0)
x = df.loc[: 'TARGET'].values
y = df.loc[:, 'TARGET'].values
print x
print y

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['TARGET']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1, 2, 3]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['TARGET'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

