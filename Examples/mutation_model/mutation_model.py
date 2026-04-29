import numpy as np

from cross_immunity_model import map_x_to_I, map_I_to_x


def new_variant_transition_function(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    jump_rate_bound=0.1,
):

    # Attention: Works only for two variants after jump
    flow_args = new_flow_args(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        beta_new,
        gamma_new,
        sigma_new,
    )

    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(0.001, x[-1])
    x[-1] = max(x[-1] - 0.001, 0)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x

    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]

    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)

    n = A.shape[0]
    zero_matrix = np.zeros((n, n))

    new_I = np.block([[A, B], [zero_matrix, C]])

    x = map_I_to_x(new_I)

    return x, flow_args


def new_variant_transition_function_with_demographics(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    old_b=0.01,
    jump_rate_bound=0.1,
):
    # Attention: Works only for two variants after jump
    flow_args = new_flow_args_with_demographics(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        old_b,
        beta_new,
        gamma_new,
        sigma_new,
    )

    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(0.001, x[-1])
    x[-1] = max(x[-1] - 0.001, 0)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x

    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]

    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)

    n = A.shape[0]
    zero_matrix = np.zeros((n, n))

    new_I = np.block([[A, B], [zero_matrix, C]])

    x = map_I_to_x(new_I)

    return x, flow_args


def new_variant_transition_function_with_matrices(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    jump_rate_bound=0.1,
    mutant_variant=-1,
    extinction_threshold=0,
):
    n = x[-1].copy()
    x = x[:-1].copy()
    # Attention: Works only for two variants after jump
    flow_args = new_flow_args_with_matrices(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        V,
        G,
        T,
        S,
        variants,
        old_b,
        beta_new,
        gamma_new,
        sigma_new,
    )

    if mutant_variant > 0:
        idx_mutant_variant = 2**mutant_variant
    else:
        idx_mutant_variant = mutant_variant
    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[idx_mutant_variant] = max(
        min(0.001, x[mutant_variant]),
        extinction_threshold + 0.1 * extinction_threshold,
    )
    x[mutant_variant] = max(x[mutant_variant] - 0.001, 0)
    x = np.array(x)

    sigma = sigma_new[0]
    protected_against_new_variant = sigma * x
    not_protected_against_new_variant = (1 - sigma) * x

    protected_against_new_variant[0] = 0
    not_protected_against_new_variant[0] = x[0]

    A = map_x_to_I(not_protected_against_new_variant)
    B = map_x_to_I(protected_against_new_variant)
    C = map_x_to_I(infectious_with_new_variant)

    zero_matrix = np.zeros_like(A)

    new_I = np.block([[A, B], [zero_matrix, C]])

    x = map_I_to_x(new_I)
    x = np.append(x, n)

    sigma_new = np.append(sigma_new, sigma_new[-1])

    transition_args = (beta_new, gamma_new, sigma_new)

    return x, flow_args, transition_args


def new_variant_transition_function_stochastic(
    t,
    x,
    alpha=10,
    beta=2,
    k=1,
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    extinction_threshold=0.05,
    V_js=np.array([[0, 1]]),
    variants_js=1,
    jump_rate_bound=0.1,
):
    I = map_x_to_I(x[:-1])
    size = 2**variants
    I_variant = (V @ I @ np.ones((size, 1))).flatten()
    I_total = np.sum(I_variant)
    mutated_variant = np.random.choice(variants, p=I_variant / I_total)

    sigma_variant = np.random.beta(alpha, beta)
    sigma_new = sigma_variant * np.array(old_sigma[mutated_variant])
    gamma_new = old_gamma[0]
    beta_new = max(
        np.random.normal(old_beta[mutated_variant], k * (1 - sigma_variant)), 0
    )

    x, flow_args, _ = new_variant_transition_function_with_matrices(
        t,
        x,
        beta_new,
        gamma_new,
        sigma_new,
        old_beta,
        old_gamma,
        old_sigma,
        V,
        G,
        T,
        S,
        variants,
        old_b,
        jump_rate_bound,
        mutated_variant,
        extinction_threshold,
    )

    jump_set_args = (
        extinction_threshold,
        flow_args[3],
        variants + 1,
    )

    if variants == 5:
        x, flow_args, jump_set_args = (
            extinction_transition_function_least_transmissible(
                t,
                x,
                *flow_args,
                extinction_threshold,
            )
        )

    return x, flow_args, jump_set_args


def new_flow_args(
    t, x, old_beta, old_gamma, old_sigma, beta=0.18, gamma=0.1, sigma=[0.5]
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros(
        (n + 1, n + 1)
    )  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1

    return beta_new, gamma_new, sigma_new


def new_flow_args_with_matrices(
    t,
    x,
    old_beta,
    old_gamma,
    old_sigma,
    V,
    G,
    T,
    S,
    n,
    old_b=0,
    beta=0.18,
    gamma=0.1,
    sigma=[0.5],
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    sigma = np.array(sigma)
    old_sigma = np.array(old_sigma)
    sigma = sigma.reshape(-1, 1)
    sigma_new = np.block([[old_sigma, sigma], [sigma.T, np.array([[1]])]])

    V_new = np.block(
        [
            [V, V],
            [np.zeros((1, V.shape[1])), np.ones((1, V.shape[1]))],
        ]
    )
    G_new = np.block([[G, gamma * np.eye(len(G))], [np.zeros_like(G), G]])
    T_new = [
        np.block(
            [
                [(1 - sigma[i]) * T[i], sigma[i] * T[i]],
                [np.zeros_like(T[i]), T[i]],
            ]
        )
        for i in range(len(T))
    ]
    T_n = construct_transition_matrix_n(sigma_new[n])
    T_new.append(T_n)
    T_new = np.array(T_new)
    n += 1
    S_new = np.array([shift_matrix(2**n, 2**m) for m in range(n)])

    return beta_new, gamma_new, sigma_new, V_new, G_new, T_new, S_new, n, old_b


def new_flow_args_with_demographics(
    t,
    x,
    old_beta,
    old_gamma,
    old_sigma,
    old_b,
    beta=0.18,
    gamma=0.1,
    sigma=[0.5],
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros(
        (n + 1, n + 1)
    )  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1

    return beta_new, gamma_new, sigma_new, old_b


def new_variant_transition_function_with_total_infections(
    t,
    x,
    beta_new=0.18,
    gamma_new=0.1,
    sigma_new=[0.5],
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    variants=1,
    jump_rate_bound=0.1,
):
    # Attention: Works only for two variants after jump
    new_flow_args = new_flow_args_with_total_infections(
        t,
        x,
        old_beta,
        old_gamma,
        old_sigma,
        variants,
        beta_new,
        gamma_new,
        sigma_new,
    )

    n = x[-variants:].copy()
    x = x[:-variants].copy()
    x = np.array(x)

    infectious_with_new_variant = np.zeros_like(x)
    infectious_with_new_variant[-1] = min(0.001, x[-1])
    x[-1] = max(x[-1] - 0.001, 0)

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


def new_flow_args_with_total_infections(
    t,
    x,
    old_beta,
    old_gamma,
    old_sigma,
    old_n,
    beta=0.18,
    gamma=0.1,
    sigma=[0.5],
):
    # Attention: Works with default arguments for two variants after jump
    beta_new = np.append(old_beta, beta)
    gamma_new = np.append(old_gamma, gamma)
    n = len(sigma)
    sigma = np.array(sigma)
    sigma_new = np.zeros(
        (n + 1, n + 1)
    )  # Initialisiere eine (n+1)x(n+1) Matrix mit Nullen

    # Fülle die Matrix
    sigma_new[:n, :n] = old_sigma  # Fülle die obere linke n x n Matrix
    sigma_new[:n, n] = sigma  # Füge sigma als letzte Spalte hinzu
    sigma_new[n, :n] = sigma.T  # Füge sigma als letzte Zeile hinzu
    sigma_new[n, n] = 1  # Setze die untere rechte Ecke auf 1

    return (beta_new, gamma_new, sigma_new, old_n + 1)


def mutation_probability_function(t, x, m=0.1):
    if len(x) < 5:
        mutation_prob = m * x[2]
    else:
        mutation_prob = 0
    return mutation_prob


def mutation_probability_function_multiple_mutations(t, x, mutation_rate=0.1):

    I = map_x_to_I(x[:-1])
    infectious = np.sum(I[1:])
    mutation_probability = mutation_rate * infectious

    return mutation_probability


def mutation_probability_function_multiple_mutations_no_dn(
    t, x, mutation_rate=0.1
):

    I = map_x_to_I(x)
    infectious = np.sum(I[1:])
    mutation_probability = mutation_rate * infectious

    return mutation_probability


def extinction_function(
    t, x, extinction_threshold=0.0005, V=np.array([[0, 1]]), variants=1
):

    if variants > 1:
        size = 2**variants
        I = map_x_to_I(x[:-1])
        I_variant = (V @ I @ np.ones((size, 1))).flatten()

        distance_to_threshold = np.min(I_variant) - extinction_threshold
    else:
        distance_to_threshold = 1

    return distance_to_threshold


def extinction_transition_function(
    t,
    x,
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    extinction_threshold=0.005,
    V_js=np.array([[0, 1]]),
    variants_js=1,
    jump_rate_bound=0.1,
):

    size = 2**variants
    I = map_x_to_I(x[:-1])
    I_variant = (V @ I @ np.ones((size, 1))).flatten()

    extinct_variant = np.argmin(I_variant)

    beta_new = np.delete(old_beta, extinct_variant)
    gamma_new = np.delete(old_gamma, extinct_variant)
    sigma_new = np.delete(
        np.delete(old_sigma, extinct_variant, axis=0), extinct_variant, axis=1
    )
    half_size = size // 2
    V_new = V[: variants - 1, :half_size]
    sets_with_extinct_variant = V[extinct_variant] == 0
    G_new = G[np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)]
    T_new = np.delete(T, extinct_variant, axis=0)
    T_new = np.array(
        [
            np.array(T_i)[
                np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)
            ]
            + np.array(T_i)[
                np.ix_(sets_with_extinct_variant, ~sets_with_extinct_variant)
            ]
            for T_i in T_new
        ]
    )
    S_new = np.array(
        [shift_matrix(2 ** (variants - 1), 2**m) for m in range(variants - 1)]
    )
    flow_args = (
        beta_new,
        gamma_new,
        sigma_new,
        V_new,
        G_new,
        T_new,
        S_new,
        variants - 1,
        old_b,
    )

    I_new = I[sets_with_extinct_variant] + I[~sets_with_extinct_variant]
    I_new = (
        I_new[:, sets_with_extinct_variant]
        + I_new[:, ~sets_with_extinct_variant]
    )
    x_new = map_I_to_x(I_new)
    x_new = np.append(x_new, x[-1])

    jump_set_args = (extinction_threshold, V_new, variants - 1)

    return x_new, flow_args, jump_set_args


def extinction_transition_function_least_transmissible(
    t,
    x,
    old_beta=[0.1],
    old_gamma=[0.1],
    old_sigma=[[1]],
    V=np.array([[0, 1]]),
    G=np.array([[0, 0.1], [0, 0]]),
    T=np.array([[[0, 1], [0, 0]]]),
    S=np.array([[[0, 0], [1, 0]]]),
    variants=1,
    old_b=0,
    extinction_threshold=0.005,
    V_js=np.array([[0, 1]]),
    variants_js=1,
    jump_rate_bound=0.1,
):

    size = 2**variants
    I = map_x_to_I(x[:-1])
    I_variant = (V @ I @ np.ones((size, 1))).flatten()

    r0 = np.array(old_beta) / np.array(old_gamma)
    extinct_variant = np.argmin(r0)

    beta_new = np.delete(old_beta, extinct_variant)
    gamma_new = np.delete(old_gamma, extinct_variant)
    sigma_new = np.delete(
        np.delete(old_sigma, extinct_variant, axis=0), extinct_variant, axis=1
    )
    half_size = size // 2
    V_new = V[: variants - 1, :half_size]
    sets_with_extinct_variant = V[extinct_variant] == 0
    G_new = G[np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)]
    T_new = np.delete(T, extinct_variant, axis=0)
    T_new = np.array(
        [
            np.array(T_i)[
                np.ix_(sets_with_extinct_variant, sets_with_extinct_variant)
            ]
            + np.array(T_i)[
                np.ix_(sets_with_extinct_variant, ~sets_with_extinct_variant)
            ]
            for T_i in T_new
        ]
    )
    S_new = np.array(
        [shift_matrix(2 ** (variants - 1), 2**m) for m in range(variants - 1)]
    )
    flow_args = (
        beta_new,
        gamma_new,
        sigma_new,
        V_new,
        G_new,
        T_new,
        S_new,
        variants - 1,
        old_b,
    )

    I_new = I[sets_with_extinct_variant] + I[~sets_with_extinct_variant]
    I_new = (
        I_new[:, sets_with_extinct_variant]
        + I_new[:, ~sets_with_extinct_variant]
    )
    x_new = map_I_to_x(I_new)
    x_new = np.append(x_new, x[-1])

    jump_set_args = (extinction_threshold, V_new, variants - 1)

    return x_new, flow_args, jump_set_args


new_variant_transition_function.returns_flow_args = True
new_variant_transition_function_with_demographics.returns_flow_args = True
new_variant_transition_function_with_matrices.returns_flow_args = True
new_variant_transition_function_stochastic.returns_flow_args = True

new_variant_transition_function_stochastic.returns_jump_set_args = True

new_variant_transition_function_with_matrices.returns_jump_time_transition_args = (
    True
)
new_variant_transition_function_with_total_infections.returns_flow_args = True
new_variant_transition_function_with_total_infections.returns_jump_rate_bound = (
    True
)
extinction_transition_function.returns_flow_args = True
extinction_transition_function.returns_jump_set_args = True
