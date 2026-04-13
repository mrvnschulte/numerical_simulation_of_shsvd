import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from cross_immunity_model import (
    multi_variant_model,
    multi_variant_model_for_loops,
    multi_variant_model_with_matrices,
    variant_in_set_matrix,
    compute_gamma_matrix,
    construct_transition_matrix,
    shift_matrix,
)
import timeit


def run_model_matrices_step():
    solve_ivp(
        multi_variant_model,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma),
    )


def run_model_for_loops():
    solve_ivp(
        multi_variant_model_for_loops,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma),
    )


def run_model_matrices_once():
    V = variant_in_set_matrix(n_variants)
    G = compute_gamma_matrix(n_variants, gamma)
    T = construct_transition_matrix(n_variants, sigma)
    S = np.array([shift_matrix(size, 2**m) for m in range(n_variants)])

    solve_ivp(
        multi_variant_model_with_matrices,
        (t0, tf),
        x0,
        t_eval=np.arange(tf),
        args=(beta, gamma, sigma, V, G, T, S, n_variants, compute_dn),
    )


def build_base_params(n_variants):
    size = 2**n_variants
    x0 = 0.001 * np.ones(3**n_variants)
    x0[0] = 1 - np.sum(x0[1:])
    beta = np.random.uniform(0.15, 0.3, n_variants)
    gamma = 0.1 * np.ones(n_variants)
    sigma = 0.8 * np.ones((size, size))
    for i in range(size):
        sigma[i, i] = 1
    compute_dn = False
    return x0, beta, gamma, sigma, compute_dn, size


np.random.seed(42)
t0 = 0
tf = 500

n = 20
n_variants_max = 7
dpi = 500

means_model_a = []
means_model_b = []
means_model_c = []
variants = list(range(1, n_variants_max + 1))

for n_variants in variants:
    x0, beta, gamma, sigma, compute_dn, size = build_base_params(n_variants)

    time_a = timeit.repeat(
        "run_model_for_loops()", globals=globals(), repeat=n, number=1
    )
    time_b = timeit.repeat(
        "run_model_matrices_step()", globals=globals(), repeat=n, number=1
    )
    time_c = timeit.repeat(
        "run_model_matrices_once()", globals=globals(), repeat=n, number=1
    )
    means_model_a.append(np.mean(time_a))
    means_model_b.append(np.mean(time_b))
    means_model_c.append(np.mean(time_c))

    # print("Model A Durchlaufzeiten (s):", np.mean(time_a))
    # print("Model B Durchlaufzeiten (s):", np.mean(time_b))
    # print("Model C Durchlaufzeiten (s):", np.mean(time_c))

plt.figure(figsize=(8, 6))
plt.plot(variants, means_model_a, label="Loops", marker="o")
plt.plot(variants, means_model_b, label="Matrices per step", marker="s")
plt.plot(variants, means_model_c, label="Matrices once", marker="^")
plt.xlabel("Number of variants")
plt.ylabel("Mean runtime [s]")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/Runtime_comparison.jpg", dpi=dpi)
plt.show()
