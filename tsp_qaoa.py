
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
from qiskit.visualization import plot_histogram


# Gloabal _lambda variable
_LAMBDA = 1

# returns the bit index for an alpha and j
def bit(i_city, l_time, num_cities):
    return i_city * num_cities + l_time 


# e^(cZZ) 
def append_zz_term(qc, q_i, q_j, gamma, constant_term):
    qc.cx(q_i, q_j)
    qc.rz(2*gamma*constant_term,q_j)
    qc.cx(q_i, q_j)


# e^(cZ)
def append_z_term(qc, q_i, gamma, constant_term):
    qc.rz(2*gamma*constant_term, q_i)

# e^(cX)
def append_x_term(qc,qi,beta):
    qc.rx(2*beta, qi)

def get_not_edge_in(G):
    N = G.number_of_nodes()
    not_edge = []
    for i in range(N):
        for j in range(N):
            if i != j:
                buffer_tupla = (i,j)
                in_edges = False
                for edge_i, edge_j in G.edges():             
                    if ( buffer_tupla == (edge_i, edge_j) or buffer_tupla == (edge_j, edge_i)):
                        in_edges = True
                if in_edges == False:
                    not_edge.append((i, j))
    return not_edge


def get_classical_simplified_z_term(G, _lambda):
    
    # recorrer la formula Z con datos grafo se va guardando en diccionario que acumula si coinciden los terminos
    N = G.number_of_nodes()
    E = G.edges()

    #  z term #

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

    # fourth term
    not_edge = get_not_edge_in(G) # include order tuples ej = (1,0), (0,1)
    for edge in not_edge:
        for l in range(N):
            i = edge[0]
            j = edge[1]
            # z_il
            z_il_index =  bit(i, l, N)
            z_classic_term[z_il_index] +=  _lambda / 4
            # z_j(l+1)
            l_plus = (l+1) % N
            z_jlplus_index =  bit(j, l_plus, N)
            z_classic_term[z_jlplus_index] +=  _lambda / 4
    
    # fifthy term
    weights = nx.get_edge_attributes(G,'weight')
    for edge_i, edge_j in G.edges(): 
        weight_ij = weights.get((edge_i,edge_j))
        weight_ji = weight_ij
        for l in range(N):

            # z_il
            z_il_index =  bit(edge_i, l, N)
            z_classic_term[z_il_index] +=  weight_ij / 4

            # z_jlplus
            l_plus = (l+1) % N
            z_jlplus_index =  bit(edge_j, l_plus, N)
            z_classic_term[z_jlplus_index] +=  weight_ij / 4

            # add order term because G.edges() do not include order tuples #
            # z_i'l
            z_il_index =  bit(edge_j, l, N)
            z_classic_term[z_il_index] +=  weight_ji / 4

            # z_j'lplus
            l_plus = (l+1) % N
            z_jlplus_index =  bit(edge_i, l_plus, N)
            z_classic_term[z_jlplus_index] +=  weight_ji / 4

    return z_classic_term


def get_classical_simplified_zz_term(G, _lambda):

    # recorrer la formula Z con datos grafo se va guardando en diccionario que acumula si coinciden los terminos
    N = G.number_of_nodes()
    E = G.edges()

    # zz term #
    zz_classic_term = [[0] * N**2 for i in range(N**2) ]
     
    # first term
    for l in range(N):
        for j in range(N):
            for i in range(N):
                if i < j:
                    # z_il
                    z_il_index =  bit(i, l, N)
                    # z_jl
                    z_jl_index =  bit(j, l, N)
                    zz_classic_term[z_il_index][z_jl_index] +=  _lambda / 2 

    # second term
    for i in range(N):
        for l in range(N):
            for j in range(N):
                if l < j:
                    # z_il
                    z_il_index =  bit(i, l, N)
                    # z_ij
                    z_ij_index =  bit(i, j, N)
                    zz_classic_term[z_il_index][z_ij_index] +=  _lambda / 2 
    
    # third term
    not_edge = get_not_edge_in(G)
    for edge in not_edge:
        for l in range(N):
            i = edge[0]
            j = edge[1]
            # z_il
            z_il_index =  bit(i, l, N)
            # z_j(l+1)
            l_plus = (l+1) % N
            z_jlplus_index =  bit(j, l_plus, N)
            zz_classic_term[z_il_index][z_jlplus_index] +=  _lambda / 4
    
    # fourth term
    weights = nx.get_edge_attributes(G,'weight')
    for edge_i, edge_j in G.edges(): 
        weight_ij = weights.get((edge_i,edge_j))
        weight_ji = weight_ij
        for l in range(N):

            # z_il
            z_il_index =  bit(edge_i, l, N)

            # z_jlplus
            l_plus = (l+1) % N
            z_jlplus_index =  bit(edge_j, l_plus, N)
            zz_classic_term[z_il_index][z_jlplus_index] +=  weight_ij / 4

            # add order term because G.edges() do not include order tuples #
            # z_i'l
            z_il_index =  bit(edge_j, l, N)

            # z_j'lplus
            l_plus = (l+1) % N
            z_jlplus_index =  bit(edge_i, l_plus, N)
            zz_classic_term[z_il_index][z_jlplus_index] +=  weight_ji / 4
    
    return zz_classic_term

    
def get_classical_simplified_hamiltonian(G, _lambda):    
    
    # z term #
    z_classic_term = get_classical_simplified_z_term(G, _lambda)
    # zz term #
    zz_classic_term = get_classical_simplified_zz_term(G, _lambda)

    return z_classic_term, zz_classic_term




def get_cost_circuit(G, gamma, _lambda):

    N =  G.number_of_nodes()
    N_square = N**2

    qc = QuantumCircuit(N_square,N_square)
      
    z_classic_term, zz_classic_term = get_classical_simplified_hamiltonian(G, _lambda)      
    
    # z term
    for i in range(N_square):
        if z_classic_term[i] != 0:
            append_z_term(qc, i, gamma, z_classic_term[i])
    
    # zz term
    for i in range(N_square):
        for j in range(N_square):
            if zz_classic_term[i][j] != 0:
                append_zz_term(qc, i, j, gamma, zz_classic_term[i][j])
    
    return qc

def get_mixer_operator(G,beta):
    
    N = G.number_of_nodes()
    qc = QuantumCircuit(N**2,N**2)
    
    for n in range(N**2):
        append_x_term(qc, n, beta)

    return qc

def invert_counts(counts):
    return {k[::-1] :v for k,v in counts.items()}

def get_QAOA_circuit(G, beta, gamma, _lambda):
    
    assert(len(beta)==len(gamma))
    
    N = G.number_of_nodes()
    qc = QuantumCircuit(N**2,N**2)
    # init min mix state
    qc.h(range(N**2))
    p = len(beta)
    
    for i in range(p):
        qc = qc.compose(get_cost_circuit(G, gamma[i], _lambda))
        qc = qc.compose(get_mixer_operator(G, beta[i]))
        qc.barrier(range(N**2))
    
    qc.measure(range(N**2),range(N**2))
    
    return qc

def tsp_obj(x, G):
    # obtenemos el valor evaluado en f(x_1, x_2,... x_n)
    z_classic_term, zz_classic_term = get_classical_simplified_hamiltonian(G, _LAMBDA)
    cost = 0

    # z term 
    for index in range(len(x)):
        z = (int(x[index]) * 2 ) -1
        cost += z_classic_term[index] * z

    ## zz term
    for i in range(len(x)):
        z_1 = (int(x[i]) * 2 ) -1
        for j in range(len(x)):
            z_2 = (int(x[j]) * 2 ) -1
            cost += zz_classic_term[i][j] * z_1 * z_1
    
    return cost


# Sample expectation value
def compute_tsp_energy(counts, G):
    energy = 0
    get_counts = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = tsp_obj(meas, G)
        energy += obj_for_meas*meas_count
        total_counts += meas_count
    return energy/total_counts

# Sample expectation value
def compute_tsp_energy_2(counts, G):
    energy = 0
    get_counts = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = tsp_obj_2(meas, G, _LAMBDA)
        energy += obj_for_meas*meas_count
        total_counts += meas_count
    return energy/total_counts

def tsp_obj_2(x, G,_lambda):
    # obtenemos el valor evaluado en f(x_1, x_2,... x_n)
    not_edge = get_not_edge_in(G)
    N = G.number_of_nodes()
    
    tsp_cost=0
    #Distancia
    weights = nx.get_edge_attributes(G,'weight')
    for edge_i, edge_j in G.edges(): 
        weight_ij = weights.get((edge_i,edge_j))
        weight_ji = weight_ij
        for l in range(N):

            # x_il
            x_il_index = bit(edge_i, l, N)

            # x_jlplus
            l_plus = (l+1) % N
            x_jlplus_index = bit(edge_j, l_plus, N)
            tsp_cost+= int(x[x_il_index]) * int(x[x_jlplus_index]) * weight_ij

            # add order term because G.edges() do not include order tuples #
            # x_i'l
            x_il_index =  bit(edge_j, l, N)

            # x_j'lplus
            x_jlplus_index =  bit(edge_i, l_plus, N)
            tsp_cost += int(x[x_il_index]) * int(x[x_jlplus_index]) * weight_ji

    #Constraint 1
    for l in range(N):
        penal1 = 1
        for i in range(N):
            x_il_index = bit(i, l, N)
            penal1 -= int(x[x_il_index])
    
        tsp_cost += _lambda * penal1**2
        
    #Contstraint 2
    for i in range(N):
        penal2 = 1
        for l in range(N):
            x_il_index = bit(i, l, N)

            penal2 -= int(x[x_il_index])
    
        tsp_cost += _lambda*penal2**2
    
    #Constraint 3    
        
    for edge in not_edge:
        for l in range(N):
            i = edge[0]
            j = edge[1]
            # x_il
            x_il_index =  bit(i, l, N)
            # x_j(l+1)
            l_plus = (l+1) % N
            x_jlplus_index = bit(j, l_plus, N)
            tsp_cost += int(x[x_il_index]) * int(x[x_jlplus_index]) * _lambda

    return tsp_cost

def get_black_box_objective(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        _lambda = _LAMBDA # get global _lambda 
        qc = get_QAOA_circuit(G, beta, gamma, _LAMBDA)
        counts = execute(qc, backend, seed_simulator=10).result().get_counts()
        return compute_tsp_energy(invert_counts(counts),G)
    return f

def get_black_box_objective_2(G,p):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        _lambda = _LAMBDA # get global _lambda 
        qc = get_QAOA_circuit(G, beta, gamma, _LAMBDA)
        counts = execute(qc, backend, seed_simulator=10).result().get_counts()
        return compute_tsp_energy_2(invert_counts(counts),G)
    return f

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
    print("labels")
    labels = nx.get_edge_attributes(G,'weight')

    #z_term, zz_term = get_classical_simplified_hamiltonian(G, 1)
    
    #print("z term")
    #print(z_term)
    #print("*****************")
    #print("zz term")
    #print(zz_term)

    #print(get_QAOA_circuit(G, beta = [2,3], gamma = [4,5], _lambda = 1))

    p = 5
    obj = get_black_box_objective(G,p)
    init_point = np.array([0.8,2.2,0.83,2.15,0.37,2.4,6.1,2.2,3.8,6.1])
    #res_sample = minimize(obj, init_point,method="COBYLA",options={"maxiter":2500,"disp":True})
    #print(res_sample)

    # Marina Solutions
    obj = get_black_box_objective_2(G,p)
    #res_sample = minimize(obj, init_point,method="COBYLA",options={"maxiter":2500,"disp":True})
    #print(res_sample)

    theta_2 = [0.72685401, 2.15678239, 0.86389827, 2.19403121, 0.26916675, 2.19832144, 7.06651453, 3.20333137, 3.81301611, 6.08893568]
    theta_1 = [0.90644898, 2.15994212, 1.8609325 , 2.14042604, 1.49126214, 2.4127999 , 6.10529434, 2.18238732, 3.84056674, 6.07097744]

    beta = theta_1[:p]
    gamma = theta_1[p:]
    _lambda = _LAMBDA # get global _lambda 
    qc = get_QAOA_circuit(G, beta, gamma, _LAMBDA)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend)
    plot_histogram(job.result().get_counts(), color='midnightblue', title="New Histogram")


    beta = theta_2[:p]
    gamma = theta_2[p:]
    _lambda = _LAMBDA # get global _lambda 
    qc = get_QAOA_circuit(G, beta, gamma, _LAMBDA)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend)
    plot_histogram(job.result().get_counts(), color='midnightblue', title="New Histogram")

    
