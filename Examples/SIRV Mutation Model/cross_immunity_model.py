# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:07:54 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def two_variant_model(t, x, beta1, beta2, sigma, gamma1=.1, gamma2=.1):
    
    i1, i12, i2, i21, s, r1, r2, r = x
    
    i1_ges = i1 + i12
    i2_ges = i2 + i21
    
    ds = -beta1 * s * i1_ges - beta2 * s * i2_ges
    di1 = (1 - sigma) * beta1 * s * i1_ges - gamma1 * i1
    dr1 = gamma1 * i1 - beta2 * r1 * i2_ges
    di21 = sigma * beta2 * s * i2_ges + beta2 * r1 * i2_ges - gamma2 * i21
    di2 = (1 - sigma) * beta2 * s * i2_ges - gamma2 * i2
    dr2 = gamma2 * i2 - beta1 * r2 * i1_ges
    di12 = sigma * beta1 * s * i1_ges + beta1 * r2 * i1_ges - gamma1 * i12
    dr = gamma2 * i21 + gamma1 * i12
    
    dx = [di1, di12, di2, di21, ds, dr1, dr2, dr]
    
    return dx

def multi_variant_model(t, x, beta, gamma, sigma):
    
    n = int(round(np.log(len(x)) / np.log(3)))
    size = 2**n

    I = map_x_to_I(x)
    
    dI = np.zeros_like(I)
    
    V = variant_in_set_matrix(n) # It could be beneficial to store these matrices in a class at their first evaluation instead of constructing it in every step
    I_variant = (V @ I @ np.ones((size, 1))).flatten()
    gamma_sum = (V.T @ gamma).flatten()
    transmission_vector = beta * I_variant
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    gamma_matrix = compute_gamma_matrix(n, gamma)
    T = construct_transition_matrix(n, sigma)
    S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    W = transmission_vector[:, np.newaxis, np.newaxis] * T
    
    for j in range(size):
        for k in range(j+1):
            if is_subset(k, j):
                dI[k, j] = (- gamma_sum[k] * I[k, j]
                            - beta_sum[j] * I[k, j]
                            + (gamma_matrix @ I)[k, j]
                            + (np.sum(S@I@W, axis=0))[k, j])
    
    dx = map_I_to_x(dI)
    
    return dx

def multi_variant_model_with_total_infections(t, x, beta, gamma, sigma, n):
    
    size = 2**n

    I = map_x_to_I(x[:3**n])
    
    dI = np.zeros_like(I)
    
    V = variant_in_set_matrix(n) # It could be beneficial to store these matrices in a class at their first evaluation instead of constructing it in every step
    I_variant = (V @ I @ np.ones((size, 1))).flatten()
    gamma_sum = (V.T @ gamma).flatten()
    transmission_vector = beta * I_variant
    beta_sum = ((np.ones_like(V) - V).T @ (transmission_vector)).flatten()
    gamma_matrix = compute_gamma_matrix(n, gamma)
    T = construct_transition_matrix(n, sigma)
    S = np.array([shift_matrix(size, 2**m) for m in range(n)])
    W = transmission_vector[:, np.newaxis, np.newaxis] * T
    S_variant = (np.ones((size, 1)).T @ I @ (np.ones_like(V) - V).T).flatten()
    dn = beta * S_variant * I_variant
    
    for j in range(size):
        for k in range(j+1):
            if is_subset(k, j):
                dI[k, j] = (- gamma_sum[k] * I[k, j]
                            - beta_sum[j] * I[k, j]
                            + (gamma_matrix @ I)[k, j]
                            + (np.sum(S@I@W, axis=0))[k, j])
    
    dx = map_I_to_x(dI)
    dx = list(dx)
    dx.extend(dn)
    
    return dx

def map_x_to_I(x):
    '''
    Maps vector representation x to matrix representation I

    Parameters
    ----------
    x : list, or np.ndarray
        Vector representation of state.

    Returns
    -------
    I : np.ndarray
        Matrix representation of state.
    '''
    
    n = int(round(np.log(len(x)) / np.log(3)))
    
    I = np.zeros((2**n, 2**n))
    
    counter = 0
    
    for j in range(2**n):
        for k in range(j+1):
            if is_subset(k, j):
                I[k, j] = x[counter]
                counter += 1

    return I

def map_I_to_x(I):
    '''
    Maps matrix representation I back to vector representation x

    Parameters
    ----------
    I : np.ndarray
        Matrix representation of state.

    Returns
    -------
    x : np.ndarray
        Vector representation of state.
    '''
    
    n = int(round(np.log2(I.shape[0])))  # Bestimme n aus der Matrixgröße
    size = 2**n  # Größe der Matrix
    
    # Berechne die Länge des Vektors x
    length_x = 3 ** n  # Anzahl der nicht-null Einträge
    x = np.zeros(length_x)  # Initialisiere den Vektor x
    
    counter = 0  # Zähler für den Vektor x
    
    # Fülle den Vektor x basierend auf der Matrix I
    for j in range(size):
        for k in range(j + 1):
            if is_subset(k, j):  # Überprüfe, ob k eine Teilmenge von j ist
                x[counter] = I[k, j]
                counter += 1
                
    return x
    
    
def is_subset(j, k):
    '''
    Checks whether j is a subset of k using binary representations.

    Parameters
    ----------
    j : int
        Binary (integer) representation of set j.
    k : int
        Binary (integer) representation of set k.

    Returns
    -------
    value : bool
        True, if j is a subset of k.
    '''

    value = (j & k == j)

    return value

def variant_in_set(i, j):
    '''
    Checks whether variant i is in set j.

    Parameters
    ----------
    i : int
        Index of variant i.
    j : int
        Binary (integer) representation of set j.

    Returns
    -------
    bool
        True, if variant i is in set j.

    '''
    return (j & (1 << (i - 1))) != 0

def variant_in_set_matrix(n):
    
    V = np.zeros((n, 2**n), dtype=int)
    
    indices = np.arange(2**n)
    
    for i in range(n):
        V[i] = (indices & (1 << i)) != 0
    
    return V

def compute_gamma_matrix(n, gamma):
    # Basisfall: A_1
    A_n = np.array([[0, gamma[0]], [0, 0]])
    
    for i in range(1, n):
        # Größe von A_n bestimmen
        size = A_n.shape[0]
        
        # Erzeuge die Einheitsmatrix I der passenden Größe
        I = np.eye(size)
        
        # Konstruiere A_{n+1}
        A_n = np.block([[A_n, gamma[i] * I], [np.zeros_like(A_n), A_n]])
    
    return A_n

def construct_transition_matrix(n, sigma):
    
    size = 2**n
    T = np.zeros((n, size, size))
    sigma = np.array(sigma)
    
    for k in range(n):
        for i in range(size):
            if not variant_in_set(k+1, i):
                for j in range(size):
                    if is_subset(i, j):
                        T[k, i, j] = 1
                        for m in range(n):
                            if variant_in_set(m+1, j-i):
                                T[k, i, j] *= sigma[k, m]
                            if not variant_in_set(m+1, j):
                                T[k, i, j] *= (1 - sigma[k, m])
                            
    return T

def shift_matrix(size, shift):
    
    return np.eye(size, k=-shift)
    
        
# n = 2
# size = 2**n
# vector_size = 3**n
# x = np.ones(vector_size) * 1/vector_size
# gamma = [.5, 1]
# beta = [1, 2]
# sigma = np.array([[1, .8], [.8, 1]])
# I = map_x_to_I(x)
# V = variant_in_set_matrix(n)
# I_variant = (V @ I @ np.ones((size, 1))).flatten()
# gamma_sum = (V.T @ gamma).flatten()
# transmission_vector = beta * I_variant
# beta_sum = ((np.ones_like(V) - V).T @ (beta * I_variant)).flatten()
# T = construct_transition_matrix(n, sigma)
# S = np.array([shift_matrix(size, 2**m) for m in range(n)])
# W = transmission_vector[:, np.newaxis, np.newaxis] * T
# S_variant = np.ones((size, 1)).T @ I @ (np.ones_like(V) - V).T



# t0 = 0
# tf = 500

# x0 = np.array([.001, 0, .001, 0, .998, 0, 0, 0])

# res12 = solve_ivp(two_variant_model, (t0, tf), x0, args=(.12, .15, .5))

# plt.plot(res12.t, res12.y.T, label=['i1', 'i12', 'i2', 'i21', 's', 'r1', 'r2', 'r'])
# plt.legend()
# plt.show()

# t0 = 0
# tf = 500

# x0 = np.array([.998, 0, .001, 0, .001, 0, 0, 0, 0])
# beta = [.2, .2]
# gamma = [.1, .15]
# sigma = np.array([[1, 0.8], [0.8, 1]])

# print(multi_variant_model(0, x0, beta, gamma, sigma))

# res12 = solve_ivp(multi_variant_model, (t0, tf), x0, t_eval=np.arange(tf), args=(beta, gamma, sigma))

# plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12'])
# plt.legend()
# plt.show()

# x0 = np.array([.999, 0, .001])
# beta = [.2]
# gamma = [.1]
# sigma = np.array([[1]])

# print(multi_variant_model(0, x0, beta, gamma, sigma))

# res1 = solve_ivp(multi_variant_model, (t0, tf), x0, t_eval=np.arange(tf), args=(beta, gamma, sigma))

# plt.plot(res1.t, res1.y.T, label=['s', 'r1', 'i1'])
# plt.legend()
# plt.show()

# x0 = np.array([.999, 0, .001])
# beta = [.2]
# gamma = [.15]
# sigma = np.array([[1]])

# print(multi_variant_model(0, x0, beta, gamma, sigma))

# res2 = solve_ivp(multi_variant_model, (t0, tf), x0, t_eval=np.arange(tf), args=(beta, gamma, sigma))

# plt.plot(res2.t, res2.y.T, label=['s', 'r1', 'i1'])
# plt.legend()
# plt.show()

t0 = 0
tf = 1000

x0 = np.array([.998, 0, .001, 0, .001, 0, 0, 0, 0])
beta = [.15, .2]
gamma = [.1, .1]
sigma = np.array([[1, 0.8], [0.8, 1]])

print(multi_variant_model(0, x0, beta, gamma, sigma))

res12 = solve_ivp(multi_variant_model, (t0, tf), x0, t_eval=np.arange(tf), args=(beta, gamma, sigma))

plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12'])
plt.legend()
plt.show()

t0 = 0
tf = 1000

r01 = 1.5
r02 = 2
g = .1


x0 = np.array([.998, 0, .001, 0, .001, 0, 0, 0, 0, .001, .001])
beta = [r01 * g, r02 * g]
gamma = [g, g]
sigma = np.array([[1, 0.8], [0.8, 1]])

print(multi_variant_model(0, x0, beta, gamma, sigma))

res12 = solve_ivp(multi_variant_model_with_total_infections, (t0, tf), x0, t_eval=np.arange(tf), args=(beta, gamma, sigma, len(beta)))

plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12', 'n1', 'n2'])
plt.legend()
plt.show()