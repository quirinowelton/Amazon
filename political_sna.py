#%%
from igraph import Graph, plot
import igraph 
import pandas as pd
import numpy as np


# Ler arquivo GML
grafo = igraph('polbooks.gml', format='gml')

# Visualizar informações básicas
print(f"Número de vértices: {grafo.vcount()}") #Visualizar o numero de vértices numero de linhas (ID por exemplo)
print(f"Número de arestas: {grafo.ecount()}") #Arestas são os links 
print(f"É direcionado? {grafo.is_directed()}") #Ver se é direcionado
#%%

plot(grafo, bbox = (1000, 800)) #Ver Grafo dos dados
#%%
#executa e define as comunidades
comunidade_ceb = grafo.community_edge_betweenness()

#%%
#Todos os testes para ver as modularidades

# In[12]: Tabela sumarizando resultados dos diferentes métodos
# Lista de métodos de detecção de comunidade
metodos = {
    "Edge Betweenness": grafo.community_edge_betweenness().as_clustering(),
    "Fast Greedy": grafo.community_fastgreedy().as_clustering(),
    "Walktrap": grafo.community_walktrap().as_clustering(),
    "Louvain": grafo.community_multilevel(),
    "Label Propagation": grafo.community_label_propagation(),
    "Spin Glass": grafo.community_spinglass()
}

# Inicializando a tabela com resultados
resultados = []

# Loop através dos métodos e calcular modularidade, número de comunidades e estatísticas dos tamanhos dos grupos
for metodo_nome, metodo in metodos.items():
    modularidade = metodo.modularity
    num_comunidades = len(metodo)
    
    # Obter o tamanho das comunidades (número de nós em cada comunidade)
    tamanhos_comunidades = [len(comunidade) for comunidade in metodo]
    
    # Calcular as estatísticas: mínimo, máximo, média e desvio padrão do tamanho das comunidades
    min_tam = np.min(tamanhos_comunidades)
    max_tam = np.max(tamanhos_comunidades)
    media_tam = np.mean(tamanhos_comunidades)
    dp_tam = np.std(tamanhos_comunidades)
    
    # Adicionar resultados na lista
    resultados.append([
        metodo_nome, 
        modularidade, 
        num_comunidades, 
        min_tam, 
        max_tam, 
        media_tam, 
        dp_tam
    ])

# Criar um DataFrame para exibir os resultados
df_resultados = pd.DataFrame(resultados, columns=[
    "Método", 
    "Modularidade", 
    "Número de Comunidades", 
    "Min Tamanho", 
    "Max Tamanho", 
    "Média Tamanho", 
    "Desvio Padrão Tamanho"
])

# In[13]: Mostrar a tabela de resultados
print(df_resultados)

#%%
import matplotlib.pyplot as plt
import numpy as np

# Exemplo: vamos usar centralidade de betweenness para dimensionar os nós
betweenness = grafo.betweenness()
sizes = [50 + 500 * (b / max(betweenness)) for b in betweenness]  
# 50 = tamanho mínimo, 500 = fator de escala

for metodo_nome, metodo in metodos.items():
    layout = grafo.layout("fr")  
    coords = np.array(layout.coords)  # posições dos nós
    cores = [metodo.membership[v] for v in range(grafo.vcount())]

    plt.figure(figsize=(8, 6))

    # Desenha arestas
    for edge in grafo.es:
        src, tgt = edge.tuple
        plt.plot([coords[src][0], coords[tgt][0]],
                 [coords[src][1], coords[tgt][1]],
                 color="lightgray", zorder=1)

    # Desenha vértices com tamanho proporcional à betweenness
    plt.scatter(coords[:,0], coords[:,1], 
                c=cores, cmap="tab20", s=sizes, zorder=2, edgecolors="k")

    plt.title(f"Comunidades detectadas por {metodo_nome}")
    plt.axis("off")
    plt.show()



