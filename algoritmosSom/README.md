######## algoritmoSom.py ########
# O algoritmoSom executa o treinamento de todos os valores pre-definidos
# e armazena todos os erros de Quantização por arquivo, emprimindo-os ordenamente em seguida 
# É importante que haja a estruta de pastas necessárias para armazenar os mapas treinados, nesse caso:
# só basta haver uma pasta chamada 'trained-model', na mesma pasta do arquivo.
# Para executalo basta executar o algoritmo sem nenhum parametro
# Exemplo: python algoritmoSom.py

######## posProcessamento.py ########
# O posProcessamento visa mostrar a a matrix U e o mapa de calor do modelo passado
# Para executar o código é necessário que seja passado 5 paramentros
# Exemplo: python algoritmoSom.py {fonte de dados(corpus do modelo)} {pesos dos modelos já treinados} {dimensao/lado do modelo treinado} {função de vizinhança do modelo treinado} {taxa de aprendizado do modelo treinado}
# Exemplo: python algoritmoSom.py 'corpora/process/som/process-binary-stemmer-corte_som.pkl' 'trained-models/process/som/process-binary-stemmer-corte_som(dim_19-sigma_2-step_0.2-epocas_2000).npy' 19 2 0.2

# Autores
# Caroline M. Buckwisser Nº USP: 9390281
# Francisco P. N. Neto Nº USP: 9862839
# Gabriel A. D. Oliveira Nº USP: 9845235
# Leonardo M Fontes Nº USP: 9019390
# Vinícius S. K. Graciliano Nº USP: 9862972