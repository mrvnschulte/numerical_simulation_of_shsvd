"""
Plot and numerically validate stability regions of a hybrid oscillator.

The undamped oscillator alternates between two modes:

* Oscillating: state [x, v], with x' = v and v' = -x.
* Held: state [x], with x' = 0.

At each prescribed jump, the oscillator changes modes. When it is released,
its velocity is set to c*x. Each oscillating phase and each held phase has
duration T.

The analytical stability region is plotted for different values of T and c.
For numerical validation, the hybrid system is simulated at sampled parameter
points using solve_hsvd from numerical_simulation_of_shsvd.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from numerical_simulation_of_shsvd import solve_hsvd


# ---------------------------------------------------------------------------
# Plot and output settings
# ---------------------------------------------------------------------------

plot_folder = Path("plots")
fileformat = "pdf"
dpi = 1200

fontsize_text = 16
fontsize_ticks = 14
alpha_grid = 0.7

plot_legend_outside = False
save_plot = True
show_plot = True

lower_limit = -20.0
upper_limit = 20.0


# ---------------------------------------------------------------------------
# Numerical validation settings
# ---------------------------------------------------------------------------

n_validation_points = 100
n_cycles = 30
growth_tolerance = 0.02
seed = 42


# ---------------------------------------------------------------------------
# Analytical boundaries
# ---------------------------------------------------------------------------


def cot(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the cotangent of the supplied times.

    Parameters
    ----------
    t : np.ndarray
        Times at which to evaluate the cotangent.

    Returns
    -------
    np.ndarray
        Cotangent values.
    """
    return -np.tan(t + np.pi / 2.0)


# ---------------------------------------------------------------------------
# Hybrid oscillator model
# ---------------------------------------------------------------------------


def oscillator_flow(
    t: float,
    y: NDArray[np.float64],
) -> list[float]:
    """
    Compute the flow of the hybrid harmonic oscillator.

    A one-dimensional state represents the held oscillator. A
    two-dimensional state represents its position and velocity.

    Parameters
    ----------
    t : float
        Current time. Not used by this autonomous system.
    y : np.ndarray
        Current state, with one or two components.

    Returns
    -------
    list of float
        State derivative.

    Raises
    ------
    ValueError
        If the state dimension is neither one nor two.
    """
    if len(y) == 1:
        return [0.0]

    if len(y) == 2:
        return [float(y[1]), -float(y[0])]

    raise ValueError(f"Unexpected state dimension: {len(y)}")


def toggle_oscillator(
    t: float,
    y: NDArray[np.float64] | list[float],
    push_constant: float,
) -> list[float]:
    """
    Release or hold the oscillator at a prescribed jump time.

    When released, the velocity is set to c times the current position.
    When held, the velocity is removed from the state.

    Parameters
    ----------
    t : float
        Current time. Not used by this transition.
    y : np.ndarray or list of float
        State immediately before the jump.
    push_constant : float
        Proportionality constant c for the release velocity.

    Returns
    -------
    list of float
        State after the jump.

    Raises
    ------
    ValueError
        If the state dimension is neither one nor two.
    """
    if len(y) == 1:
        position = float(y[0])
        return [position, push_constant * position]

    if len(y) == 2:
        return [float(y[0])]

    raise ValueError(f"Unexpected state dimension: {len(y)}")


def simulate_oscillator(
    flow_time: float,
    push_constant: float,
    n_cycles: int,
) -> float:
    """
    Simulate repeated release-and-hold cycles.

    The oscillator starts held at position 1 and is released at time zero.
    Each cycle consists of an oscillating phase of length T followed by a
    held phase of length T.

    Parameters
    ----------
    flow_time : float
        Duration T of each oscillating and held phase. Must be positive.
    push_constant : float
        Proportionality constant c for the release velocity.
    n_cycles : int
        Number of complete cycles to simulate. Must be positive.

    Returns
    -------
    float
        Absolute position after the final held phase.

    Raises
    ------
    ValueError
        If flow_time or n_cycles is not positive.
    """
    if flow_time <= 0:
        raise ValueError("flow_time must be positive.")

    if n_cycles <= 0:
        raise ValueError("n_cycles must be positive.")

    final_time = 2.0 * n_cycles * flow_time

    # The first jump releases the oscillator at time zero. The final jump
    # holds it at the start of the last held phase.
    jump_times = [index * flow_time for index in range(2 * n_cycles)]

    trajectory = solve_hsvd(
        flow_function=oscillator_flow,
        initial_value=[1.0],
        t_span=(0.0, final_time),
        jump_times=jump_times,
        jump_time_transition_function=toggle_oscillator,
        jump_time_transition_args=(push_constant,),
        t_eval=[0.0, final_time],
        rtol=1e-9,
        atol=1e-12,
    )

    return abs(float(trajectory(final_time)[0]))


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------


def validate_stability_region(
    t_min: float,
    t_max: float,
    c_min: float,
    c_max: float,
    n_points: int,
    n_cycles: int,
    growth_tolerance: float,
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """
    Compare simulated trajectories with the analytical stability region.

    Parameter points are sampled uniformly from the specified rectangle.
    Points close to the poles of the cotangent are excluded.

    The numerical classification is based on the average logarithmic change
    in the absolute position over the simulated cycles. Points whose growth
    rates lie within the tolerance are classified as inconclusive.

    Parameters
    ----------
    t_min : float
        Lower bound for the sampled flow time T.
    t_max : float
        Upper bound for the sampled flow time T.
    c_min : float
        Lower bound for the sampled push constant c.
    c_max : float
        Upper bound for the sampled push constant c.
    n_points : int
        Number of parameter points to simulate.
    n_cycles : int
        Number of release-and-hold cycles per simulation.
    growth_tolerance : float
        Tolerance around zero for the average logarithmic change per cycle.
    seed : int
        Seed for reproducible parameter sampling.

    Returns
    -------
    validation_t : np.ndarray
        Sampled flow times.
    validation_c : np.ndarray
        Sampled push constants.
    numerically_stable : np.ndarray
        Boolean mask for points classified as asymptotically stable.
    numerically_unstable : np.ndarray
        Boolean mask for points classified as unstable.
    numerically_inconclusive : np.ndarray
        Boolean mask for points whose classification is inconclusive.
    disagreements : np.ndarray
        Boolean mask for conclusive points where simulation and analytical
        classifications disagree.

    Notes
    -----
    A finite simulation does not prove asymptotic stability. In particular,
    classifications close to the boundary depend on the number of cycles,
    the tolerance, and numerical integration accuracy.
    """
    rng = np.random.default_rng(seed)

    sampled_t: list[float] = []
    sampled_c: list[float] = []

    while len(sampled_t) < n_points:
        candidate_t = float(rng.uniform(t_min, t_max))
        candidate_c = float(rng.uniform(c_min, c_max))

        # Avoid numerical difficulties near multiples of pi.
        if abs(np.sin(candidate_t)) < 1e-3:
            continue

        sampled_t.append(candidate_t)
        sampled_c.append(candidate_c)

    validation_t = np.asarray(sampled_t, dtype=float)
    validation_c = np.asarray(sampled_c, dtype=float)

    final_positions = np.array(
        [
            simulate_oscillator(flow_time, push_constant, n_cycles)
            for flow_time, push_constant in zip(validation_t, validation_c)
        ],
        dtype=float,
    )

    # The initial position is 1. Clipping prevents taking the logarithm of zero.
    growth_rates = (
        np.log(np.maximum(final_positions, np.finfo(float).tiny)) / n_cycles
    )

    numerically_stable = growth_rates < -growth_tolerance
    numerically_unstable = growth_rates > growth_tolerance
    numerically_inconclusive = ~(numerically_stable | numerically_unstable)

    cot_values = cot(validation_t)
    upper_limits = -cot_values + np.sqrt(cot_values**2 + 1.0)
    lower_limits = -cot_values - np.sqrt(cot_values**2 + 1.0)

    analytically_stable = (validation_c > lower_limits) & (
        validation_c < upper_limits
    )

    conclusive = ~numerically_inconclusive
    disagreements = conclusive & (numerically_stable != analytically_stable)

    print("\nNumerical validation using solve_hsvd")
    print(f"  Parameter points: {n_points}")
    print(f"  Cycles per point: {n_cycles}")
    print(f"  Classified as asymptotically stable: {numerically_stable.sum()}")
    print(f"  Classified as unstable: {numerically_unstable.sum()}")
    print(f"  Inconclusive: {numerically_inconclusive.sum()}")
    print(f"  Disagreements with analytical region: {disagreements.sum()}")

    if np.any(disagreements):
        print("  Disagreeing points (T, c, final absolute position):")
        for flow_time, push_constant, final_position in zip(
            validation_t[disagreements],
            validation_c[disagreements],
            final_positions[disagreements],
        ):
            print(
                f"    T={flow_time:.6f}, c={push_constant:.6f}, "
                f"|x_final|={final_position:.6e}"
            )

    return (
        validation_t,
        validation_c,
        numerically_stable,
        numerically_unstable,
        numerically_inconclusive,
        disagreements,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Plot the analytical region and overlay numerical classifications.

    The figure is optionally saved and displayed according to the global
    plot settings.
    """
    # Flow times used for the analytical boundaries.
    times = np.linspace(0.01, 3.0, 601) * np.pi

    cot_values = cot(times)
    upper_c = -cot_values + np.sqrt(cot_values**2 + 1.0)
    lower_c = -cot_values - np.sqrt(cot_values**2 + 1.0)

    (
        validation_t,
        validation_c,
        numerically_stable,
        numerically_unstable,
        numerically_inconclusive,
        disagreements,
    ) = validate_stability_region(
        t_min=float(times[0]),
        t_max=float(times[-1]),
        c_min=lower_limit,
        c_max=upper_limit,
        n_points=n_validation_points,
        n_cycles=n_cycles,
        growth_tolerance=growth_tolerance,
        seed=seed,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Analytical boundaries and stability regions.
    ax.plot(times, upper_c, color="red", label="Stability boundary")
    ax.plot(times, lower_c, color="red")
    ax.fill_between(
        times,
        lower_c,
        upper_c,
        color="blue",
        alpha=0.2,
        label="Asymptotically stable",
    )
    ax.fill_between(
        times,
        lower_limit,
        lower_c,
        color="red",
        alpha=0.2,
        label="Unstable",
    )
    ax.fill_between(
        times,
        upper_c,
        upper_limit,
        color="red",
        alpha=0.2,
    )

    # Overlay classifications based on simulated trajectories.
    ax.scatter(
        validation_t[numerically_stable],
        validation_c[numerically_stable],
        color="blue",
        marker=".",
        s=20,
        label="Simulation: asymptotically stable",
    )
    ax.scatter(
        validation_t[numerically_unstable],
        validation_c[numerically_unstable],
        color="darkred",
        marker="x",
        s=20,
        label="Simulation: unstable",
    )
    ax.scatter(
        validation_t[numerically_inconclusive],
        validation_c[numerically_inconclusive],
        color="black",
        marker="o",
        s=20,
        label="Simulation: inconclusive",
    )

    if np.any(disagreements):
        ax.scatter(
            validation_t[disagreements],
            validation_c[disagreements],
            facecolors="none",
            edgecolors="black",
            s=100,
            linewidths=1.5,
            label="Disagreement",
        )

    # Label the horizontal axis in multiples of pi/2.
    n_ticks = int(np.round(times[-1] / np.pi * 2)) + 1
    xticks = [index * np.pi / 2 for index in range(n_ticks)]
    x_tick_labels = [
        f"{index // 2}π" if index % 2 == 0 else f"{index}π/2"
        for index in range(n_ticks)
    ]
    x_tick_labels[0] = "0"
    x_tick_labels[1] = "π/2"
    x_tick_labels[2] = "π"

    ax.set_xticks(xticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlim(0.0, times[-1])
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_xlabel(r"Flow time $T$", fontsize=fontsize_text)
    ax.set_ylabel(r"Push constant $c$", fontsize=fontsize_text)
    ax.set_title(
        "Stability region of the hold-and-push oscillator",
        fontsize=fontsize_text,
    )
    ax.tick_params(axis="both", labelsize=fontsize_ticks)
    ax.grid(True, alpha=alpha_grid)

    if plot_legend_outside:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=fontsize_ticks,
        )
    else:
        ax.legend(fontsize=fontsize_ticks)

    fig.tight_layout()

    if save_plot:
        plot_folder.mkdir(parents=True, exist_ok=True)
        output_path = plot_folder / f"stability_region.{fileformat}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {output_path.resolve()}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
