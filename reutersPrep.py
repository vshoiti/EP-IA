#!/usr/bin/env python
import sys, re

# invocado com: python reutersPrep.py [arquivo entrada] [arquivo saida]
if (len(sys.argv) < 3):
  print 'missing file names'
  sys.exit()

readFile = sys.argv[1]
writeFile = sys.argv[2]

# TRECHO ABAIXO RETIRADO DE: https://stackoverflow.com/questions/3277503/how-do-i-read-a-file-line-by-line-into-a-list
with open(readFile) as f: 
  content = f.readlines() # le as linhas do arquivo e armazena em content
  
content = [x.strip(' \n') for x in content] # remove espacos e '\n's no inicio e fim de cada string
# FIM DO TRECHO RETIRADO DE: https://stackoverflow.com/questions/3277503/how-do-i-read-a-file-line-by-line-into-a-list
    
# loop que procura as tags de inicio de texto e concatena as 
# linhas que encontra apos a tag em uma linha no arquivo f
with open(writeFile, 'w') as f:
  isWriting = False # flag que determina se a proxima linha sera concatenada a saida
  for s in content:
    if '<BODY>' in s: # se contem a tag <BODY>  
      isWriting = True # comeca a concatenar
      s = re.sub(r'.*<BODY>', '', s) # remove de s a tag <BODY> e tudo que vem antes

    if '&#3;</BODY></TEXT>' in s: # se contem a tag de final de texto
      s = re.sub(r'&#3;</BODY></TEXT>.*', '', s) # remove de s a tag &#3;</BODY></TEXT> e tudo que vem depois
      f.write(s) # escreve o resto de s em f
      f.write('\n\n') # pula uma linha
      isWriting = False # para de escrever
      
    if isWriting:
      f.write(s) # concatena s ao arquivo f
      f.write(' ')

