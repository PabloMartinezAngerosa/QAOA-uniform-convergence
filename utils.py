import networkx as nx

# genera un grafo networkx desde matriz de pesos y conexiones
def mapeo_grafo(matriz_conexiones,matriz_pesos):
    G=nx.Graph()
    for i, filas in enumerate(matriz_conexiones):
        G.add_node(i, pos=(i,i))
        for j, elemento_conexion in enumerate(filas):
            if elemento_conexion == 1:
                elemento_peso = matriz_pesos[i][j]
                G.add_edge(i,j,weight=elemento_peso)
                print(elemento_peso)
    return G
