
from thirdParty.classical import rand_graph, classical, bitstring_to_path, calc_cost
import numpy as np
from utils import mapeo_grafo

import qiskit
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import op_converter
from qiskit.aqua.operators import WeightedPauliOperator


# returns the bit index for an alpha and j
def bit(alpha, j, num_cities):
    return j * num_cities + alpha

# e^(cZZ) #TODO: terminar de adaptar
def append_zz_term(qc,q1,q2,gamma):
    qc.cx(q1,q2)
    qc.rz(2*gamma,q2)
    qc.cx(q1,q2)

# e^(cZ) #TODO: terminar de adaptar
def append_z_term():
    pass


def get_classical_simplified_hamiltonian(G, _lambda):

    # recorrer la formula Z con datos grafo se va guardando en diccionario que acumula si coinciden los terminos
    N = G.number_of_nodes()
    E = G.edges()

    # Z 

    z_classic_term = [0] * N**2

    # first term
    for l in range(N):
        for i in range(N):
            z_il_index =  bit(i, l, N)
            z_classic_term[z_il_index] += -1 * _lambda

    # second term
    for l in range(N):
        for j in range(N):
            for i in range(N):
                if i < j:
                    # z_il
                    z_il_index =  bit(i, l, N)
                    z_classic_term[z_il_index] +=  _lambda / 2
                    # z_jl
                    z_jl_index =  bit(j, l, N)
                    z_classic_term[z_jl_index] +=  _lambda / 2 

    # third term
    for i in range(N):
        for l in range(N):
            for j in range(N):
                if l < j:
                    # z_il
                    z_il_index =  bit(i, l, N)
                    z_classic_term[z_il_index] +=  _lambda / 2
                    # z_ij
                    z_ij_index =  bit(i, j, N)
                    z_classic_term[z_ij_index] +=  _lambda / 2

    # third term
    not_edge = []
    for i in range(N):
        for j in range(N):
            dummy_tupla = (i,j)
            for edge_i, edge_j in G.edges():
                if ( dummy_tupla != (edge_i, edge_j) and dummy_tupla != (edge_j, edge_i)):
                    not_edge.append(dummy_tupla)


    

    # recorre la formula ZZ con datos grafo se va guardando en diccionario que acumula si coinciden los terminos
    # retorna Z, ZZ



def get_cost_circuit(G, gamma, _lambda):

    N =  G.number_of_nodes()
    qc = QuantumCircuit(N^2,N^2)
    
    #TODO optimizamos formula 
    # llamamos una funcion que retorna dos 2 diccionarios con z term y zz term en clasico. 
    z_classic_term, zz_classic_term = get_classical_simplified_hamiltonian(G, _lambda)  

    #TODO recorremos los diccianarios terminos y arma el circuito cuantico. 
    
    for i,j in G.edges():
        append_zz_term(qc,i,j,gamma)
    return qc



if __name__ == '__main__':
    
    cantidad_ciudades = 4
    pesos, conexiones = None, None
    mejor_camino = None

    while not mejor_camino:
        pesos, conexiones = rand_graph(cantidad_ciudades)
        mejor_costo, mejor_camino = classical(pesos, conexiones, loop=False)

    G = mapeo_grafo(conexiones, pesos)

    print(mejor_costo)
    print(mejor_camino)
    print(G.edges())
    print(G.nodes())
