import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from cross_immunity_model import (
    cases_to_incidence,
    construct_matrices,
    create_initial_conditions_from_seroprevalence,
    cumulative_cases_to_incidence,
    cumulative_to_daily_cases,
    multi_variant_model_with_matrices,
    relative_to_absolute_cases,
    map_x_to_I,
    infectious_per_variant,
)

import pandas as pd


def read_data(filename: str) -> pd.DataFrame:
    """Read incidence data from a CSV file."""
    df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
    return df


def params_to_sigma(sigma_params: np.ndarray, n: int) -> list[list[float]]:
    """Reconstruct a symmetric matrix with 1 on the diagonal from upper-triangular values.

    Args:
        sigma_params: Flattened upper-triangular off-diagonal elements.
        n: Matrix dimension.

    Returns:
        Symmetric n x n matrix with 1 on diagonal.
    """
    sigma = np.eye(n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            sigma[i, j] = sigma_params[idx]
            sigma[j, i] = sigma_params[idx]
            idx += 1
    return sigma.tolist()


def run_model(
    beta: list[float],
    gamma: list[float],
    sigma: list[list[float]],
    infectious_at_start: list[float],
    fraction_antibodies: float,
    antibody_effectiveness: list[float],
    t0: int,
    tf: int,
    population: int,
    b: float,
) -> np.ndarray:
    """Run the ODE model and return the modelled incidence.

    Returns:
        Array of shape (tf - t0 - 1, n_variants) with 7-day incidence per 100,000.
    """
    n_variants = len(beta)
    t_eval = np.arange(t0, tf)

    x0 = create_initial_conditions_from_seroprevalence(
        fraction_antibodies, antibody_effectiveness, infectious_at_start
    )
    n0 = np.zeros(n_variants)
    V, G, T, S = construct_matrices(gamma, sigma)

    y0 = np.append(x0, n0)
    sol = solve_ivp(
        multi_variant_model_with_matrices,
        (t0, tf),
        y0,
        t_eval=t_eval,
        args=(beta, gamma, sigma, V, G, T, S, b, True),
    )

    cumulative_cases = sol.y.T[:, -n_variants:]
    incidence = cumulative_cases_to_incidence(cumulative_cases, population)
    return incidence


def objective(
    params: np.ndarray,
    gamma: list[float],
    fraction_antibodies: float,
    antibody_effectiveness: list[float],
    t0: int,
    tf: int,
    population: int,
    b: float,
    reported_incidence: np.ndarray,
    n_variants: int,
) -> float:
    """Compute the sum of squared errors between model and reported incidence.

    Args:
        params: Flattened parameter vector [infectious_at_start (log), beta, sigma_upper].
        ...

    Returns:
        Sum of squared residuals.
    """
    # Unpack parameters
    log_infectious = params[:n_variants]
    infectious_at_start = (10**log_infectious).tolist()

    beta = params[n_variants : 2 * n_variants].tolist()

    n_sigma_params = n_variants * (n_variants - 1) // 2
    sigma_params = params[2 * n_variants : 2 * n_variants + n_sigma_params]
    sigma = params_to_sigma(sigma_params, n_variants)

    # Run model
    try:
        incidence = run_model(
            beta,
            gamma,
            sigma,
            infectious_at_start,
            fraction_antibodies,
            antibody_effectiveness,
            t0,
            tf,
            population,
            b,
        )
    except Exception:
        return 1e12

    # Align lengths (reported might be shorter or longer)
    n_points = min(len(incidence), len(reported_incidence))
    residuals = incidence[:n_points] - reported_incidence[:n_points]

    return np.nansum(residuals**2)


# --- Configuration ---
filename_incidence = "D:/Projects/Dissertation/Assignment of Cases to Variants/data/moving_average_per_variant.csv"
start_date = "2021-11-01"
variants = ["Delta"]  # , "Omicron BA.1", "Omicron BA.2"]

t0 = 0
tf = 210
n_variants = 3
population = 83497147
b = 1e-5

# Fixed parameters
fraction_antibodies = 0.891
antibody_effectiveness = [0.88, 0.5, 0.5]
gamma = [0.15, 0.2, 0.2]

# --- Read reported data ---
df_reported = read_data(filename_incidence)
start = pd.Timestamp(start_date)
end_date = start + pd.Timedelta(days=tf - 1)
df_reported_slice = df_reported.loc[start:end_date, variants]
reported_incidence = cases_to_incidence(df_reported_slice.values, population)

# --- Initial guesses ---
infectious_at_start_init = [0.0001, 1e-12, 1e-15]
beta_init = [0.9, 0.8, 0.8]
sigma_upper_init = [0.8, 0.8, 0.8]  # σ₁₂, σ₁₃, σ₂₃

# Pack into parameter vector (use log10 for infectious_at_start)
p0 = np.concatenate(
    [
        np.log10(
            infectious_at_start_init
        ),  # log-scale for numerical stability
        beta_init,
        sigma_upper_init,
    ]
)

# --- Bounds ---
n_sigma_params = n_variants * (n_variants - 1) // 2
bounds = (
    # log10(infectious_at_start): e.g. 1e-20 to 1e-1
    [(-20, -1)] * n_variants
    # beta: positive, reasonable range
    + [(0.1, 5.0)] * n_variants
    # sigma off-diagonal: between 0 and 1
    + [(0.0, 1.0)] * n_sigma_params
)

# --- Modell mit Anfangswerten ausführen ---
init_infectious = (10 ** p0[:n_variants]).tolist()
init_beta = p0[n_variants : 2 * n_variants].tolist()
init_sigma = params_to_sigma(p0[2 * n_variants :], n_variants)

incidence_init = run_model(
    init_beta,
    gamma,
    init_sigma,
    init_infectious,
    fraction_antibodies,
    antibody_effectiveness,
    t0,
    tf,
    population,
    b,
)

# --- Plot ---
dates_ode = pd.date_range(start=start, periods=len(incidence_init), freq="D")

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10.colors

for i, variant in enumerate(variants):
    ax.plot(
        df_reported_slice.index,
        reported_incidence[:, i],
        label=f"{variant} (reported)",
        color=colors[i],
        marker="o",
        markersize=2,
        linestyle="none",
    )
    ax.plot(
        dates_ode,
        incidence_init[:, i],
        label=f"{variant} (Anfangswerte)",
        color=colors[i],
        linestyle="-",
    )

ax.set_xlabel("Date")
ax.set_ylabel("7-day incidence per 100,000")
ax.set_title("Gemeldete Inzidenz vs. Modell mit Anfangswerten")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# --- Optimization ---
# print("Starting optimization...")
# result = minimize(
#     objective,
#     p0,
#     args=(
#         gamma,
#         fraction_antibodies,
#         antibody_effectiveness,
#         t0,
#         tf,
#         population,
#         b,
#         reported_incidence,
#         n_variants,
#     ),
#     method="Nelder-Mead",
#     bounds=bounds,
#     options={"maxiter": 1000, "disp": True},
# )
# print(f"Optimization success: {result.success}")
# print(f"Message: {result.message}")

# # --- Extract optimized parameters ---
# opt_params = result.x
# opt_infectious = (10 ** opt_params[:n_variants]).tolist()
# opt_beta = opt_params[n_variants : 2 * n_variants].tolist()
# opt_sigma = params_to_sigma(opt_params[2 * n_variants :], n_variants)

# print(f"\nOptimized parameters:")
# print(f"  infectious_at_start: {opt_infectious}")
# print(f"  beta:                {opt_beta}")
# print(f"  sigma:\n{np.array(opt_sigma)}")

# # --- Run model with optimized parameters ---
# incidence_opt = run_model(
#     opt_beta,
#     gamma,
#     opt_sigma,
#     opt_infectious,
#     fraction_antibodies,
#     antibody_effectiveness,
#     t0,
#     tf,
#     population,
#     b,
# )

# # --- Plot comparison ---
# dates_ode = pd.date_range(start=start, periods=len(incidence_opt), freq="D")

# fig, ax = plt.subplots(figsize=(12, 6))
# colors = plt.cm.tab10.colors

# for i, variant in enumerate(variants):
#     ax.plot(
#         df_reported_slice.index,
#         reported_incidence[:, i],
#         label=f"{variant} (reported)",
#         color=colors[i],
#         marker="o",
#         markersize=2,
#         linestyle="none",
#     )
#     ax.plot(
#         dates_ode,
#         incidence_opt[:, i],
#         label=f"{variant} (model)",
#         color=colors[i],
#         linestyle="-",
#     )

# ax.set_xlabel("Date")
# ax.set_ylabel("7-day incidence per 100,000")
# ax.set_title("Reported vs. optimized model incidence by variant")
# ax.legend()
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show(
# --- Feste Parameter (nicht optimiert) ---

sigma_fixed = params_to_sigma(sigma_upper_init, n_variants)
infectious_fixed = infectious_at_start_init  # fix


def objective_delta(params: np.ndarray) -> float:
    """Optimiert beta[0], gamma[0], infectious_at_start[0] und antibody_effectiveness[0] (Delta)."""
    beta_delta, gamma_delta, log_inf_delta, ab_eff_delta = params

    beta_opt = [beta_delta] + beta_init[1:]
    gamma_opt = [gamma_delta] + gamma[1:]
    infectious_opt = [10**log_inf_delta] + infectious_fixed[1:]
    ab_eff_opt = [ab_eff_delta] + antibody_effectiveness[1:]  # nur [0] frei

    try:
        incidence = run_model(
            beta_opt,
            gamma_opt,
            sigma_fixed,
            infectious_opt,
            fraction_antibodies,
            ab_eff_opt,
            t0,
            tf,
            population,
            b,
        )
    except Exception:
        return 1e12

    n_points = min(len(incidence), len(reported_incidence))
    residuals = incidence[:n_points, 0] - reported_incidence[:n_points, 0]
    return np.nansum(residuals**2)


# --- Startwerte & Bounds ---
p0_delta = np.array(
    [
        beta_init[0],  # beta[0]
        gamma[0],  # gamma[0]
        np.log10(infectious_at_start_init[0]),  # log10(infectious[0])
        antibody_effectiveness[0],  # antibody_effectiveness[0]
    ]
)

bounds_delta = [
    (0.1, 5.0),  # beta_delta
    (0.05, 0.5),  # gamma_delta
    (-20, -1),  # log10(infectious_at_start[0])
    (0.0, 1.0),  # antibody_effectiveness[0]
]

# --- Optimierung ---
print("Starte Optimierung (Delta: beta, gamma, infectious_at_start)...")
result = minimize(
    objective_delta,
    p0_delta,
    method="Nelder-Mead",
    bounds=bounds_delta,
    options={"maxiter": 2000, "disp": True},
)
print(f"Erfolg: {result.success} | {result.message}")

opt_beta_delta, opt_gamma_delta, opt_log_inf_delta, opt_ab_eff_delta = result.x
opt_inf_delta = 10**opt_log_inf_delta

print(f"\nOptimierte Parameter (Delta):")
print(f"  beta[0]               = {opt_beta_delta:.4f}")
print(f"  gamma[0]              = {opt_gamma_delta:.4f}")
print(f"  infectious_at_start[0]= {opt_inf_delta:.2e}")
print(f"  antibody_effectiveness[0] = {opt_ab_eff_delta:.4f}")

# Modell mit optimierten Werten
beta_opt = [opt_beta_delta] + beta_init[1:]
gamma_opt = [opt_gamma_delta] + gamma[1:]
infectious_opt = [opt_inf_delta] + infectious_fixed[1:]
ab_eff_opt = [opt_ab_eff_delta] + antibody_effectiveness[1:]

incidence_opt = run_model(
    beta_opt,
    gamma_opt,
    sigma_fixed,
    infectious_opt,
    fraction_antibodies,
    ab_eff_opt,
    t0,
    tf,
    population,
    b,
)

# --- Plot ---
dates_ode = pd.date_range(start=start, periods=len(incidence_opt), freq="D")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    df_reported_slice.index,
    reported_incidence[:, 0],
    label="Delta (gemeldet)",
    color="steelblue",
    marker="o",
    markersize=2,
    linestyle="none",
)
ax.plot(
    dates_ode,
    incidence_init[:, 0],
    label="Delta (Anfangswerte)",
    color="gray",
    linestyle="--",
)
ax.plot(
    dates_ode,
    incidence_opt[:, 0],
    label=f"Delta (optimiert: β={opt_beta_delta:.3f}, γ={opt_gamma_delta:.3f}, I₀={opt_inf_delta:.1e})",
    color="steelblue",
    linestyle="-",
)
ax.set_xlabel("Datum")
ax.set_ylabel("7-Tage-Inzidenz pro 100.000")
ax.set_title("Delta: Optimierung von β, γ und infectious_at_start")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# --- BA.1-Daten einlesen ---
variants_ba1 = ["Delta", "Omicron BA.1"]
df_reported_ba1 = df_reported.loc[start:end_date, variants_ba1]
reported_incidence_ba1 = cases_to_incidence(df_reported_ba1.values, population)


# --- Stage 2: Delta (re-optimiert) + BA.1 + sigma_12 ---
def objective_ba1(params: np.ndarray) -> float:
    """Optimiert alle Delta-Parameter (Stage 1 als Startwert) +
    beta[1], gamma[1], infectious[1], ab_eff[1], sigma_12."""
    (
        beta_delta2,
        gamma_delta2,
        log_inf_delta2,
        ab_eff_delta2,
        beta_ba1,
        gamma_ba1,
        log_inf_ba1,
        ab_eff_ba1,
        sigma_12,
    ) = params

    beta_opt2 = [beta_delta2, beta_ba1, beta_init[2]]
    gamma_opt2 = [gamma_delta2, gamma_ba1, gamma[2]]
    infectious_opt2 = [
        10**log_inf_delta2,
        10**log_inf_ba1,
        infectious_fixed[2],
    ]
    ab_eff_opt2 = [ab_eff_delta2, ab_eff_ba1, antibody_effectiveness[2]]

    sigma_upper_opt2 = [sigma_12, sigma_upper_init[1], sigma_upper_init[2]]
    sigma_opt2 = params_to_sigma(sigma_upper_opt2, n_variants)

    try:
        incidence = run_model(
            beta_opt2,
            gamma_opt2,
            sigma_opt2,
            infectious_opt2,
            fraction_antibodies,
            ab_eff_opt2,
            t0,
            tf,
            population,
            b,
        )
    except Exception:
        return 1e12

    n_points = min(len(incidence), len(reported_incidence_ba1))
    # Residuen auf Delta (0) UND BA.1 (1)
    residuals = (
        incidence[:n_points, :2] - reported_incidence_ba1[:n_points, :2]
    )
    return np.nansum(residuals**2)


# --- Startwerte: Delta aus Stage 1, BA.1 aus Initialwerten ---
p0_ba1 = np.array(
    [
        # Delta – Startwerte aus Stage-1-Ergebnis
        opt_beta_delta,
        opt_gamma_delta,
        opt_log_inf_delta,  # bereits log10-skaliert
        opt_ab_eff_delta,
        # BA.1 – initiale Schätzwerte
        beta_init[1],
        gamma[1],
        np.log10(infectious_at_start_init[1]),
        antibody_effectiveness[1],
        # sigma_12
        sigma_upper_init[0],
    ]
)

bounds_ba1 = [
    # Delta
    (0.1, 5.0),  # beta_delta
    (0.05, 0.5),  # gamma_delta
    (-20, -1),  # log10(infectious[0])
    (0.0, 1.0),  # ab_eff[0]
    # BA.1
    (0.1, 5.0),  # beta_ba1
    (0.05, 0.5),  # gamma_ba1
    (-20, -1),  # log10(infectious[1])
    (0.0, 1.0),  # ab_eff[1]
    # sigma
    (0.0, 1.0),  # sigma_12
]

print("Starte Stage-2-Optimierung (Delta re-fit + BA.1 + sigma_12)...")
result_ba1 = minimize(
    objective_ba1,
    p0_ba1,
    method="Nelder-Mead",
    bounds=bounds_ba1,
    options={"maxiter": 5000, "disp": True},
)
print(f"Erfolg: {result_ba1.success} | {result_ba1.message}")

(
    opt_beta_delta2,
    opt_gamma_delta2,
    opt_log_inf_delta2,
    opt_ab_eff_delta2,
    opt_beta_ba1,
    opt_gamma_ba1,
    opt_log_inf_ba1,
    opt_ab_eff_ba1,
    opt_sigma_12,
) = result_ba1.x

opt_inf_delta2 = 10**opt_log_inf_delta2
opt_inf_ba1 = 10**opt_log_inf_ba1

print(f"\nOptimierte Parameter (Stage 2):")
print(f"  --- Delta (re-fit) ---")
print(
    f"  beta[0]                   = {opt_beta_delta2:.4f}  (Stage 1: {opt_beta_delta:.4f})"
)
print(
    f"  gamma[0]                  = {opt_gamma_delta2:.4f}  (Stage 1: {opt_gamma_delta:.4f})"
)
print(
    f"  infectious_at_start[0]    = {opt_inf_delta2:.2e}  (Stage 1: {opt_inf_delta:.2e})"
)
print(
    f"  antibody_effectiveness[0] = {opt_ab_eff_delta2:.4f}  (Stage 1: {opt_ab_eff_delta:.4f})"
)
print(f"  --- BA.1 ---")
print(f"  beta[1]                   = {opt_beta_ba1:.4f}")
print(f"  gamma[1]                  = {opt_gamma_ba1:.4f}")
print(f"  infectious_at_start[1]    = {opt_inf_ba1:.2e}")
print(f"  antibody_effectiveness[1] = {opt_ab_eff_ba1:.4f}")
print(f"  --- Cross-Immunity ---")
print(f"  sigma_12 (Delta↔BA.1)     = {opt_sigma_12:.4f}")


# --- Modell mit Stage-2-Parametern ---
sigma_upper_s2 = [opt_sigma_12, sigma_upper_init[1], sigma_upper_init[2]]

incidence_opt2 = run_model(
    [opt_beta_delta2, opt_beta_ba1, beta_init[2]],
    [opt_gamma_delta2, opt_gamma_ba1, gamma[2]],
    params_to_sigma(sigma_upper_s2, n_variants),
    [opt_inf_delta2, opt_inf_ba1, infectious_fixed[2]],
    fraction_antibodies,
    [opt_ab_eff_delta2, opt_ab_eff_ba1, antibody_effectiveness[2]],
    t0,
    tf,
    population,
    b,
)

# --- Plot ---
dates_ode2 = pd.date_range(start=start, periods=len(incidence_opt2), freq="D")
variant_labels = ["Delta", "Omicron BA.1"]
colors_plot = ["steelblue", "darkorange"]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for i, (label, color) in enumerate(zip(variant_labels, colors_plot)):
    ax = axes[i]
    ax.plot(
        df_reported_ba1.index,
        reported_incidence_ba1[:, i],
        label=f"{label} (gemeldet)",
        color=color,
        marker="o",
        markersize=2,
        linestyle="none",
    )
    # Stage-1-Ergebnis nur für Delta zum Vergleich
    if i == 0:
        ax.plot(
            dates_ode,
            incidence_opt[:, 0],
            label="Delta (Stage 1)",
            color="gray",
            linestyle="--",
        )
    ax.plot(
        dates_ode2,
        incidence_opt2[:, i],
        label=f"{label} (Stage 2)",
        color=color,
        linestyle="-",
    )
    ax.set_ylabel("7-Tage-Inzidenz pro 100.000")
    ax.set_title(label)
    ax.legend()

axes[1].set_xlabel("Datum")
fig.suptitle(f"Stage 2  |  σ₁₂={opt_sigma_12:.3f}")
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# --- BA.2-Daten einlesen ---
variants_all = ["Delta", "Omicron BA.1", "Omicron BA.2"]
df_reported_all = df_reported.loc[start:end_date, variants_all]
reported_incidence_all = cases_to_incidence(df_reported_all.values, population)


# --- Stage 3: Alle Parameter re-fit + BA.2 + sigma_13, sigma_23 ---
def objective_ba2(params: np.ndarray) -> float:
    """Optimiert alle Delta- und BA.1-Parameter (Stage 2 als Startwert) +
    beta[2], gamma[2], infectious[2], ab_eff[2], sigma_13, sigma_23.
    sigma_12 wird aus Stage 2 übernommen und ebenfalls re-fit."""
    (
        beta_delta3,
        gamma_delta3,
        log_inf_delta3,
        ab_eff_delta3,
        beta_ba1_3,
        gamma_ba1_3,
        log_inf_ba1_3,
        ab_eff_ba1_3,
        beta_ba2,
        gamma_ba2,
        log_inf_ba2,
        ab_eff_ba2,
        sigma_12_3,
        sigma_13,
        sigma_23,
    ) = params

    beta_opt3 = [beta_delta3, beta_ba1_3, beta_ba2]
    gamma_opt3 = [gamma_delta3, gamma_ba1_3, gamma_ba2]
    infectious_opt3 = [10**log_inf_delta3, 10**log_inf_ba1_3, 10**log_inf_ba2]
    ab_eff_opt3 = [ab_eff_delta3, ab_eff_ba1_3, ab_eff_ba2]

    # sigma-Matrix: (0,1)=sigma_12, (0,2)=sigma_13, (1,2)=sigma_23
    sigma_upper_opt3 = [sigma_12_3, sigma_13, sigma_23]
    sigma_opt3 = params_to_sigma(sigma_upper_opt3, n_variants)

    try:
        incidence = run_model(
            beta_opt3,
            gamma_opt3,
            sigma_opt3,
            infectious_opt3,
            fraction_antibodies,
            ab_eff_opt3,
            t0,
            tf,
            population,
            b,
        )
    except Exception:
        return 1e12

    n_points = min(len(incidence), len(reported_incidence_all))
    # Residuen auf alle 3 Varianten
    residuals = incidence[:n_points, :] - reported_incidence_all[:n_points, :]
    return np.nansum(residuals**2)


# --- Startwerte: Delta + BA.1 aus Stage 2, BA.2 aus Initialwerten ---
p0_ba2 = np.array(
    [
        # Delta – Stage-2-Ergebnis
        opt_beta_delta2,
        opt_gamma_delta2,
        opt_log_inf_delta2,
        opt_ab_eff_delta2,
        # BA.1 – Stage-2-Ergebnis
        opt_beta_ba1,
        opt_gamma_ba1,
        opt_log_inf_ba1,
        opt_ab_eff_ba1,
        # BA.2 – Initialwerte
        beta_init[2],
        gamma[2],
        np.log10(infectious_at_start_init[2]),
        antibody_effectiveness[2],
        # Sigma – sigma_12 aus Stage 2, sigma_13/23 aus Initialwerten
        opt_sigma_12,
        sigma_upper_init[1],  # sigma_13
        sigma_upper_init[2],  # sigma_23
    ]
)

bounds_ba2 = [
    # Delta
    (0.1, 5.0),  # beta_delta
    (0.05, 0.5),  # gamma_delta
    (-20, -1),  # log10(infectious[0])
    (0.0, 1.0),  # ab_eff[0]
    # BA.1
    (0.1, 5.0),  # beta_ba1
    (0.05, 0.5),  # gamma_ba1
    (-20, -1),  # log10(infectious[1])
    (0.0, 1.0),  # ab_eff[1]
    # BA.2
    (0.1, 5.0),  # beta_ba2
    (0.05, 0.5),  # gamma_ba2
    (-20, -1),  # log10(infectious[2])
    (0.0, 1.0),  # ab_eff[2]
    # Sigma
    (0.0, 1.0),  # sigma_12
    (0.0, 1.0),  # sigma_13 (Delta↔BA.2)
    (0.0, 1.0),  # sigma_23 (BA.1↔BA.2)
]

print(
    "Starte Stage-3-Optimierung (alle Varianten + BA.2 + sigma_13, sigma_23)..."
)
result_ba2 = minimize(
    objective_ba2,
    p0_ba2,
    method="Nelder-Mead",
    bounds=bounds_ba2,
    options={"maxiter": 10000, "disp": True},
)
print(f"Erfolg: {result_ba2.success} | {result_ba2.message}")

(
    opt_beta_delta3,
    opt_gamma_delta3,
    opt_log_inf_delta3,
    opt_ab_eff_delta3,
    opt_beta_ba1_3,
    opt_gamma_ba1_3,
    opt_log_inf_ba1_3,
    opt_ab_eff_ba1_3,
    opt_beta_ba2,
    opt_gamma_ba2,
    opt_log_inf_ba2,
    opt_ab_eff_ba2,
    opt_sigma_12_3,
    opt_sigma_13,
    opt_sigma_23,
) = result_ba2.x

opt_inf_delta3 = 10**opt_log_inf_delta3
opt_inf_ba1_3 = 10**opt_log_inf_ba1_3
opt_inf_ba2 = 10**opt_log_inf_ba2

print(f"\nOptimierte Parameter (Stage 3):")
print(f"  --- Delta ---")
print(
    f"  beta[0]                   = {opt_beta_delta3:.4f}  (Stage 2: {opt_beta_delta2:.4f})"
)
print(
    f"  gamma[0]                  = {opt_gamma_delta3:.4f}  (Stage 2: {opt_gamma_delta2:.4f})"
)
print(
    f"  infectious_at_start[0]    = {opt_inf_delta3:.2e}  (Stage 2: {opt_inf_delta2:.2e})"
)
print(
    f"  antibody_effectiveness[0] = {opt_ab_eff_delta3:.4f}  (Stage 2: {opt_ab_eff_delta2:.4f})"
)
print(f"  --- BA.1 ---")
print(
    f"  beta[1]                   = {opt_beta_ba1_3:.4f}  (Stage 2: {opt_beta_ba1:.4f})"
)
print(
    f"  gamma[1]                  = {opt_gamma_ba1_3:.4f}  (Stage 2: {opt_gamma_ba1:.4f})"
)
print(
    f"  infectious_at_start[1]    = {opt_inf_ba1_3:.2e}  (Stage 2: {opt_inf_ba1:.2e})"
)
print(
    f"  antibody_effectiveness[1] = {opt_ab_eff_ba1_3:.4f}  (Stage 2: {opt_ab_eff_ba1:.4f})"
)
print(f"  --- BA.2 ---")
print(f"  beta[2]                   = {opt_beta_ba2:.4f}")
print(f"  gamma[2]                  = {opt_gamma_ba2:.4f}")
print(f"  infectious_at_start[2]    = {opt_inf_ba2:.2e}")
print(f"  antibody_effectiveness[2] = {opt_ab_eff_ba2:.4f}")
print(f"  --- Cross-Immunity ---")
print(
    f"  sigma_12 (Delta↔BA.1)     = {opt_sigma_12_3:.4f}  (Stage 2: {opt_sigma_12:.4f})"
)
print(f"  sigma_13 (Delta↔BA.2)     = {opt_sigma_13:.4f}")
print(f"  sigma_23 (BA.1↔BA.2)      = {opt_sigma_23:.4f}")


# --- Modell mit Stage-3-Parametern ---
sigma_upper_s3 = [opt_sigma_12_3, opt_sigma_13, opt_sigma_23]

incidence_opt3 = run_model(
    [opt_beta_delta3, opt_beta_ba1_3, opt_beta_ba2],
    [opt_gamma_delta3, opt_gamma_ba1_3, opt_gamma_ba2],
    params_to_sigma(sigma_upper_s3, n_variants),
    [opt_inf_delta3, opt_inf_ba1_3, opt_inf_ba2],
    fraction_antibodies,
    [opt_ab_eff_delta3, opt_ab_eff_ba1_3, opt_ab_eff_ba2],
    t0,
    tf,
    population,
    b,
)

# --- Plot ---
dates_ode3 = pd.date_range(start=start, periods=len(incidence_opt3), freq="D")
variant_labels = ["Delta", "Omicron BA.1", "Omicron BA.2"]
colors_plot = ["steelblue", "darkorange", "seagreen"]
stage2_results = [incidence_opt2[:, 0], incidence_opt2[:, 1], None]

fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

for i, (label, color, s2) in enumerate(
    zip(variant_labels, colors_plot, stage2_results)
):
    ax = axes[i]
    ax.plot(
        df_reported_all.index,
        reported_incidence_all[:, i],
        label=f"{label} (gemeldet)",
        color=color,
        marker="o",
        markersize=2,
        linestyle="none",
    )
    if s2 is not None:
        ax.plot(
            dates_ode2,
            s2,
            label=f"{label} (Stage 2)",
            color="gray",
            linestyle="--",
        )
    ax.plot(
        dates_ode3,
        incidence_opt3[:, i],
        label=f"{label} (Stage 3)",
        color=color,
        linestyle="-",
    )
    ax.set_ylabel("7-Tage-Inzidenz pro 100.000")
    ax.set_title(label)
    ax.legend()

axes[2].set_xlabel("Datum")
fig.suptitle(
    f"Stage 3  |  σ₁₂={opt_sigma_12_3:.3f}  σ₁₃={opt_sigma_13:.3f}  σ₂₃={opt_sigma_23:.3f}",
    fontsize=13,
)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
