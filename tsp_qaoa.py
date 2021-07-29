import qiskit
from thirdParty.classical import rand_graph, classical, bitstring_to_path, calc_cost
import networkx as nx

from utils import mapeo_grafo

if __name__ == '__main__':
    
    cantidad_ciudades = 4
    pesos, conexiones = None, None
    mejor_camino = None

    while not mejor_camino:
        pesos, conexiones = rand_graph(cantidad_ciudades)
        mejor_costo, mejor_camino = classical(pesos, conexiones, loop=False)

    #TODO adaptar grafo a el nunpy y hacer plot
    print(pesos)
    print(conexiones)
    print(mejor_camino)
    print(mejor_costo)

    G = mapeo_grafo(conexiones, pesos)
    print(G)

