
# coding: utf-8

# In[22]:


import random


# In[23]:


def loadCorpus(sourcepath):
    corpus = list()
    with open(sourcepath, 'r', encoding="ISO-8859-1") as f:
        for line in f:
            if line.strip(): # se a line nao for vazia, line.strip() == true
                corpus.append(line)
    return corpus


# In[24]:


data = loadCorpus('raw/reuters.txt')


# In[25]:


data = random.sample(data, 5500)


# In[26]:


toWrite = ''
for text in data:
    text = text.strip()
    toWrite += text + '\n'


# In[28]:


with open('raw/reuters-cortado.txt', 'w') as f:
    f.write(toWrite)

