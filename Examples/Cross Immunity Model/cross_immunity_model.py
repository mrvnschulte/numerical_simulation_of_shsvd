# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:07:54 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os
from numerical_simulation_of_shsvd import (solve_hsvd, solve_shsvd_single_run,
                                           solve_shsvd)

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

def new_variant_transition_function(t, x, sigma=.5):
    # Attention: Works only for two variants after jump
    
    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(.001, x[-1])
    x[-1] = max(x[-1]-.001, 0)

    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x
    
    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]
    
    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)
    
    n = A.shape[0]
    zero_matrix = np.zeros((n, n))

    # Kombiniere die Arrays
    new_I = np.block([[A, B], [zero_matrix, C]])
    
    x = map_I_to_x(new_I)
    
    return x

def new_flow_args(t, x, old_beta, old_gamma, old_sigma, beta=.18, gamma=.1,
                  sigma=[.5]):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros((n + 1, n + 1))  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1
    
    return beta_new, gamma_new, sigma_new


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

def new_variant_transition_function_with_total_infections(
        t, x, beta_new=.18, gamma_new=.1, sigma_new=[.5], old_beta=[.1],
        old_gamma=[.1], old_sigma=[[1]], variants=1, jump_rate_bound=.1
    ):
    # Attention: Works only for two variants after jump
    new_flow_args = new_flow_args_with_total_infections(t, x, old_beta,
                                                        old_gamma, old_sigma,
                                                        variants)

    n = x[-variants:].copy()
    x = x[:-variants].copy()
    x = np.array(x)
    
    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(.001, x[-1])
    x[-1] = max(x[-1]-.001, 0)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x
    
    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]
    
    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)
    
    size = A.shape[0]
    zero_matrix = np.zeros((size, size))

    # Kombiniere die Arrays
    new_I = np.block([[A, B], [zero_matrix, C]])
    
    x = map_I_to_x(new_I)
    n_new = np.append(n, infectious_with_new_variant[-1])
    
    x = np.append(x, n_new)
    
    return x, new_flow_args, 1e-10

def new_flow_args_with_total_infections(t, x, old_beta, old_gamma, old_sigma,
                                        old_n, beta=.18, gamma=.1, sigma=[.5]):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros((n + 1, n + 1))  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1
    
    return (beta_new, gamma_new, sigma_new, old_n+1)

def total_infections(x):
    if isinstance(x[0], (list, np.ndarray)):
        infections = [total_infections(y) for y in x]
        return infections
    else:
        infections = x[-1]
        if len(x)>5:
            infections += x[-2]
    return infections

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

# t0 = 0
# tf = 1000

# x0 = np.array([.998, 0, .001, 0, .001, 0, 0, 0, 0])
# beta = [.15, .2]
# gamma = [.1, .1]
# sigma = np.array([[1, 0.8], [0.8, 1]])

# print(multi_variant_model(0, x0, beta, gamma, sigma))

# res12 = solve_ivp(multi_variant_model, (t0, tf), x0, t_eval=np.arange(tf), args=(beta, gamma, sigma))

# plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12'])
# plt.legend()
# plt.show()

t0 = 0
tf = 1000

r01 = 1.5
r02 = 2
g = .1

file1 = 'final_infections_1.csv'
file2 = 'final_infections_2.csv'

# Überprüfen, ob die Dateien existieren
if os.path.exists(file1) and os.path.exists(file2):
    # Einlesen der CSV-Dateien
    final_infections_1 = np.loadtxt(file1, delimiter=',', skiprows=1)
    final_infections_2 = np.loadtxt(file2, delimiter=',', skiprows=1)

    # Extrahiere sigmas und r02s
    r02s = final_infections_1[:, 0]  # Die erste Spalte ist sigmas
    final_infections_1 = final_infections_1[:, 1:]  # Der Rest sind die Werte für final_infections_1
    final_infections_2 = final_infections_2[:, 1:]  # Der Rest sind die Werte für final_infections_2

    # sigmas extrahieren, indem wir die erste Zeile der CSV Datei lesen und die Kopfzeile ignorieren
    header = np.loadtxt(file1, delimiter=',', max_rows=1, dtype=str)  # Lies die Kopfzeile als String
    sigmas = header[1:].astype(float)  # Wandle die Werte in float um, überspringe das erste Element

else:
    r02s = np.linspace(1, 2, 51)
    sigmas = np.linspace(0, 1, 21)
    
    final_infections_1 = np.zeros((len(r02s), len(sigmas)))
    final_infections_2 = np.zeros((len(r02s), len(sigmas)))
    
    for i, r02 in enumerate(r02s):
        for j, s in enumerate(sigmas):
            
    
            x0 = np.array([.998, 0, .001, 0, .001, 0, 0, 0, 0, .001, .001])
            beta = [r01 * g, r02 * g]
            gamma = [g, g]
            sigma = np.array([[1, s], [s, 1]])
            
            # print(multi_variant_model(0, x0, beta, gamma, sigma))
            
            res12 = solve_ivp(multi_variant_model_with_total_infections,
                              (t0, tf), x0, t_eval=np.arange(tf),
                              args=(beta, gamma, sigma, len(beta)))
            
            final_infections_1[i, j] = res12.y[-2][-1]
            final_infections_2[i, j] = res12.y[-1][-1]
            # plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12', 'n1', 'n2'])
            # plt.legend()
            # plt.show()
        
    final_infections_1_with_r02 = np.column_stack((r02s, final_infections_1))
    final_infections_2_with_r02 = np.column_stack((r02s, final_infections_2))

    # Speichern der Matrizen als CSV-Dateien
    np.savetxt(file1, final_infections_1_with_r02, delimiter=',', header='r02,' + ','.join(map(str, sigmas)), comments='')
    np.savetxt(file2, final_infections_2_with_r02, delimiter=',', header='r02,' + ','.join(map(str, sigmas)), comments='')

c = plt.pcolormesh(sigmas, r02s, final_infections_1)
plt.colorbar(c, label='Final number of infections with variant 1')
plt.title(r'Final number of infections with variant 1 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.pcolormesh(sigmas, r02s, final_infections_2)
plt.colorbar(c, label='Final number of infections with variant 2')
plt.title(r'Final number of infections with variant 2 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.pcolormesh(sigmas, r02s, final_infections_1 + final_infections_2)
plt.colorbar(c, label='Final number of infections')
plt.title(r'Final number of infections($R_0^1$=1.5, $\gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_1)
plt.colorbar(c, label='Final number of infections with variant 1')
plt.title(r'Final number of infections with variant 1 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_2)
plt.colorbar(c, label='Final number of infections with variant 2')
plt.title(r'Final number of infections with variant 2 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_1 + final_infections_2)
plt.colorbar(c, label='Final number of infections')
plt.title(r'Final number of infections($R_0^1$=1.5, $\gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

t0 = 0
tf = 1000

r01 = 1.5
r02 = 2
g = .1

file1 = 'final_infections_1_joint_initial.csv'
file2 = 'final_infections_2_joint_initial.csv'

# Überprüfen, ob die Dateien existieren
if os.path.exists(file1) and os.path.exists(file2):
    # Einlesen der CSV-Dateien
    final_infections_1 = np.loadtxt(file1, delimiter=',', skiprows=1)
    final_infections_2 = np.loadtxt(file2, delimiter=',', skiprows=1)

    # Extrahiere sigmas und r02s
    r02s = final_infections_1[:, 0]  # Die erste Spalte ist sigmas
    final_infections_1 = final_infections_1[:, 1:]  # Der Rest sind die Werte für final_infections_1
    final_infections_2 = final_infections_2[:, 1:]  # Der Rest sind die Werte für final_infections_2

    # sigmas extrahieren, indem wir die erste Zeile der CSV Datei lesen und die Kopfzeile ignorieren
    header = np.loadtxt(file1, delimiter=',', max_rows=1, dtype=str)  # Lies die Kopfzeile als String
    sigmas = header[1:].astype(float)  # Wandle die Werte in float um, überspringe das erste Element

else:
    r02s = np.linspace(1, 2, 51)
    sigmas = np.linspace(0, 1, 21)
    
    final_infections_1 = np.zeros((len(r02s), len(sigmas)))
    final_infections_2 = np.zeros((len(r02s), len(sigmas)))
    
    for i, r02 in enumerate(r02s):
        for j, s in enumerate(sigmas):
            
    
            x0 = np.array([.999, 0, 0, 0, 0, 0, 0, 0, .001, .001, .001])
            beta = [r01 * g, r02 * g]
            gamma = [g, g]
            sigma = np.array([[1, s], [s, 1]])
            
            # print(multi_variant_model(0, x0, beta, gamma, sigma))
            
            res12 = solve_ivp(multi_variant_model_with_total_infections,
                              (t0, tf), x0, t_eval=np.arange(tf),
                              args=(beta, gamma, sigma, len(beta)))
            
            final_infections_1[i, j] = res12.y[-2][-1]
            final_infections_2[i, j] = res12.y[-1][-1]
            # plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12', 'n1', 'n2'])
            # plt.legend()
            # plt.show()
        
    final_infections_1_with_r02 = np.column_stack((r02s, final_infections_1))
    final_infections_2_with_r02 = np.column_stack((r02s, final_infections_2))

    # Speichern der Matrizen als CSV-Dateien
    np.savetxt(file1, final_infections_1_with_r02, delimiter=',', header='r02,' + ','.join(map(str, sigmas)), comments='')
    np.savetxt(file2, final_infections_2_with_r02, delimiter=',', header='r02,' + ','.join(map(str, sigmas)), comments='')

c = plt.pcolormesh(sigmas, r02s, final_infections_1)
plt.colorbar(c, label='Final number of infections with variant 1')
plt.title(r'Final number of infections with variant 1 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.pcolormesh(sigmas, r02s, final_infections_2)
plt.colorbar(c, label='Final number of infections with variant 2')
plt.title(r'Final number of infections with variant 2 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.pcolormesh(sigmas, r02s, final_infections_1 + final_infections_2)
plt.colorbar(c, label='Final number of infections')
plt.title(r'Final number of infections($R_0^1$=1.5, $\gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_1)
plt.colorbar(c, label='Final number of infections with variant 1')
plt.title(r'Final number of infections with variant 1 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_2)
plt.colorbar(c, label='Final number of infections with variant 2')
plt.title(r'Final number of infections with variant 2 ($R_0^1=1.5, \gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

c = plt.contourf(sigmas, r02s, final_infections_1 + final_infections_2)
plt.colorbar(c, label='Final number of infections')
plt.title(r'Final number of infections($R_0^1$=1.5, $\gamma_1=\gamma_2=0.1$)')
plt.xlabel(r'Cross-immunity $\sigma$')
plt.ylabel(r'$R_0^2$')
plt.show()

tf = 1000
x0 = np.array([.999, 0, .001, .001])
beta = [.15]
gamma = [.1]
sigma = [[1]]
beta_new = .2
gamma_new = .1
t_eval = np.arange(tf+1)

file1 = 'final_infections_1_delayed.csv'
file2 = 'final_infections_2_delayed.csv'

# Überprüfen, ob die Dateien existieren
if os.path.exists(file1) and os.path.exists(file2):
    # Einlesen der CSV-Dateien
    final_infections_1 = np.loadtxt(file1, delimiter=',', skiprows=1)
    final_infections_2 = np.loadtxt(file2, delimiter=',', skiprows=1)

    # Extrahiere sigmas und r02s
    cross_immunities = final_infections_1[:, 0]  # Die erste Spalte ist sigmas
    final_infections_1 = final_infections_1[:, 1:]  # Der Rest sind die Werte für final_infections_1
    final_infections_2 = final_infections_2[:, 1:]  # Der Rest sind die Werte für final_infections_2

    # sigmas extrahieren, indem wir die erste Zeile der CSV Datei lesen und die Kopfzeile ignorieren
    header = np.loadtxt(file1, delimiter=',', max_rows=1, dtype=str)  # Lies die Kopfzeile als String
    delays = header[1:].astype(float)  # Wandle die Werte in float um, überspringe das erste Element

else:
    cross_immunities = np.linspace(0, 1, 21)
    delays = np.linspace(0, 250, 51)
    
    final_infections_1 = np.zeros((len(cross_immunities), len(delays)))
    final_infections_2 = np.zeros((len(cross_immunities), len(delays)))

    for i, s in enumerate(cross_immunities):
        for j, delay in enumerate(delays):
            sigma_new = [s]
            
            func = solve_hsvd(multi_variant_model_with_total_infections, x0,
                              (t0, tf),
                              t_eval=t_eval,
                              jump_times=[delay],
                              jump_time_transition_function=
                              new_variant_transition_function_with_total_infections,
                              flow_args=(beta, gamma, sigma, 1),
                              flow_args_transition_function=
                              new_flow_args_with_total_infections,
                              flow_args_transition_args=(beta_new, gamma_new,
                                                         sigma_new),
                              jump_time_transition_args=(sigma_new[0],))
            #func.plot_coordinates(labels=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12',
             #                             'i2,12', 'i12', 'n1', 'n2'])
            
            final_infections_1[i, j] = func(tf)[-2]
            final_infections_2[i, j] = func(tf)[-1]
            # plt.plot(res12.t, res12.y.T, label=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12', 'i2,12', 'i12', 'n1', 'n2'])
            # plt.legend()
            # plt.show()
        
    final_infections_1_with_sigma = np.column_stack((cross_immunities, final_infections_1))
    final_infections_2_with_sigma = np.column_stack((cross_immunities, final_infections_2))

    # Speichern der Matrizen als CSV-Dateien
    np.savetxt(file1, final_infections_1_with_sigma, delimiter=',', header='sigma,' + ','.join(map(str, delays)), comments='')
    np.savetxt(file2, final_infections_2_with_sigma, delimiter=',', header='sigma,' + ','.join(map(str, delays)), comments='')

c = plt.pcolormesh(delays, cross_immunities, final_infections_1)
plt.colorbar(c, label='Final number of infections with variant 1')
plt.title('Final number of infections with variant 1\n' r'($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)')
plt.ylabel(r'Cross-immunity $\sigma$')
plt.xlabel(r'Time of mutation')
plt.show()

c = plt.pcolormesh(delays, cross_immunities, final_infections_2)
plt.colorbar(c, label='Final number of infections with variant 2')
plt.title('Final number of infections with variant 2\n' r'($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)')
plt.ylabel(r'Cross-immunity $\sigma$')
plt.xlabel(r'Time of mutation')
plt.show()

c = plt.pcolormesh(delays, cross_immunities, final_infections_1 + final_infections_2)
plt.colorbar(c, label='Final number of infections')
plt.title(r'Final number of infections ($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)')
plt.ylabel(r'Cross-immunity $\sigma$')
plt.xlabel(r'Time of mutation')
plt.show()

c = plt.contourf(delays, cross_immunities, final_infections_1)
plt.colorbar(c, label='Final number of infections with variant 1')
plt.title('Final number of infections with variant 1\n' r'($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)')
plt.ylabel(r'Cross-immunity $\sigma$')
plt.xlabel(r'Time of mutation')
plt.show()

c = plt.contourf(delays, cross_immunities, final_infections_2)
plt.colorbar(c, label='Final number of infections with variant 2')
plt.title('Final number of infections with variant 2\n' r'($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)')
plt.ylabel(r'Cross-immunity $\sigma$')
plt.xlabel(r'Time of mutation')
plt.show()

c = plt.contourf(delays, cross_immunities, final_infections_1 + final_infections_2)
plt.colorbar(c, label='Final number of infections')
plt.title(r'Final number of infections ($R_0^1=1.5, R_0^2=2, \gamma_1=\gamma_2=0.1$)')
plt.ylabel(r'Cross-immunity $\sigma$')
plt.xlabel(r'Time of mutation')
plt.show()

sigma_new = [.5]
new_variant_transition_function_with_total_infections.returns_flow_args = True
new_variant_transition_function_with_total_infections.returns_jump_rate_bound = True

def jump_set_function(t, x):
    if len(x) < 5:
        return x[-1] - .2
    else:
        return x[-2] - .199
    
def jump_rate_function(t, x, m=.1):
    if len(x)<5:
        return m * x[2]
    else:
        return 0
    
mutation_rate = .1

func = solve_hsvd(multi_variant_model_with_total_infections, x0,
                  (t0, tf),
                  t_eval=t_eval,
                  jump_times=[100],
                  jump_time_transition_function=
                  new_variant_transition_function_with_total_infections,
                  flow_args=(beta, gamma, sigma, 1),
                  jump_time_transition_args=(beta_new, gamma_new, sigma_new))
func.plot_coordinates(labels=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12',
                              'i2,12', 'i12', 'n1', 'n2'])
func = solve_shsvd_single_run(
    multi_variant_model_with_total_infections, 
    jump_rate_function,
    new_variant_transition_function_with_total_infections,
    x0,
    (t0, tf),
    jump_rate_bound=mutation_rate,
    t_eval=t_eval,
    #jump_times=[100],
    jump_time_transition_function=
    new_variant_transition_function_with_total_infections,
    flow_args=(beta, gamma, sigma, 1),
    jump_time_transition_args=(beta_new, gamma_new, sigma_new),
    jump_rate_args=(mutation_rate,))
func.plot_coordinates(labels=['s', 'r1', 'i1', 'r2', 'i2', 'r12', 'i1,12',
                              'i2,12', 'i12', 'n1', 'n2'])

# stats = solve_shsvd(
#     multi_variant_model_with_total_infections, 
#     jump_rate_function,
#     new_variant_transition_function_with_total_infections,
#     x0,
#     (t0, tf),
#     jump_rate_bound=mutation_rate,
#     t_eval=t_eval,
#     #jump_times=[100],
#     jump_time_transition_function=
#     new_variant_transition_function_with_total_infections,
#     flow_args=(beta, gamma, sigma, 1),
#     jump_time_transition_args=(beta_new, gamma_new, sigma_new),
#     jump_rate_args=(mutation_rate,),
#     tol=1e-2,
#     min_iterations=20,
#     convergence_estimate=('cef', 0),
#     )