import qiskit
from thirdParty.classical import rand_graph, classical, bitstring_to_path, calc_cost
import numpy as np

if __name__ == '__main__':
    
    cantidad_ciudades = 4
    pesos, conexiones = None, None
    mejor_camino = None

    while not mejor_camino:
        pesos, conexiones = rand_graph(cantidad_ciudades)
        mejor_costo, mejor_camino = classical(pesos, conexiones, loop=False)

    print(pesos)
    print(conexiones)
    print(mejor_camino)
    print(mejor_costo)
    

