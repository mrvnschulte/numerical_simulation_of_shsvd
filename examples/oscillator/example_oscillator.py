"""
Demonstrate a hybrid harmonic oscillator with variable state dimension.

The oscillator has two operating modes:

* Oscillating: state [x, v], where x is position and v is velocity.
* Held: state [x], where the position remains constant.

Transitions switch between the two modes. The example includes:

1. A deterministic simulation with prescribed jumps and a jump-set event.
2. A single stochastic trajectory with prescribed and stochastic jumps.
3. Statistics across multiple stochastic trajectories.

Install the package from the project root:

    pip install -e ".[dev]"

Run the example:

    python harmonic_oscillator.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from numerical_simulation_of_shsvd import (
    FunctionOfVariableDimension,
    solve_hsvd,
    solve_shsvd,
    solve_shsvd_single_run,
)


# ---------------------------------------------------------------------------
# Plot and output settings
# ---------------------------------------------------------------------------

plot_folder = Path("plots")
fileformat = "pdf"
dpi = 1200

fontsize_text = 16
fontsize_ticks = 14
alpha_grid = 0.7

save_plots = True
show_plots = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def oscillator_flow(
    t: float,
    y: NDArray[np.float64],
    mass: float,
    stiffness: float,
    damping: float,
) -> list[float]:
    """
    Compute the state derivative of the hybrid oscillator.

    In the oscillating mode, the state is [position, velocity] and obeys
    the damped harmonic oscillator equation. In the held mode, the state
    contains only the position, which remains constant.

    Parameters
    ----------
    t : float
        Current time. Not used by this autonomous model.
    y : np.ndarray
        Current state, with one or two components.
    mass : float
        Oscillator mass. Must be positive.
    stiffness : float
        Spring stiffness.
    damping : float
        Damping coefficient.

    Returns
    -------
    list of float
        Derivative of the state.

    Raises
    ------
    ValueError
        If the state dimension is neither one nor two.
    """
    if len(y) == 1:
        return [0.0]

    if len(y) == 2:
        position, velocity = y
        acceleration = (
            -(stiffness / mass) * position - (damping / mass) * velocity
        )
        return [float(velocity), float(acceleration)]

    raise ValueError(f"Unexpected state dimension: {len(y)}")


def toggle_oscillator(
    t: float,
    y: NDArray[np.float64] | list[float],
    release_factor: float,
) -> list[float]:
    """
    Switch between the held and oscillating modes.

    When releasing the oscillator, its velocity is set to the position
    multiplied by release_factor. When holding it, the velocity is removed
    from the state.

    Parameters
    ----------
    t : float
        Current time. Not used by this transition.
    y : np.ndarray or list of float
        Current state, with one or two components.
    release_factor : float
        Factor used to calculate the velocity when releasing the oscillator.

    Returns
    -------
    list of float
        State after the transition.

    Raises
    ------
    ValueError
        If the state dimension is neither one nor two.
    """
    if len(y) == 1:
        position = float(y[0])
        return [position, release_factor * position]

    if len(y) == 2:
        return [float(y[0])]

    raise ValueError(f"Unexpected state dimension: {len(y)}")


def jump_rate(
    t: float,
    y: NDArray[np.float64] | list[float],
) -> float:
    """
    Compute the stochastic jump rate.

    The rate is the absolute position, capped at one. Consequently, one
    is a valid upper bound for all states.

    Parameters
    ----------
    t : float
        Current time. Not used by this rate function.
    y : np.ndarray or list of float
        Current state. Its first component is the position.

    Returns
    -------
    float
        Jump rate between zero and one.
    """
    return min(1.0, abs(float(y[0])))


def lower_position_event(
    t: float,
    y: NDArray[np.float64],
    threshold: float,
) -> float:
    """
    Define an event when the position reaches a threshold.

    The direction attribute configured below restricts event detection to
    downward crossings.

    Parameters
    ----------
    t : float
        Current time. Not used by this event function.
    y : np.ndarray
        Current state. Its first component is the position.
    threshold : float
        Position at which the event occurs.

    Returns
    -------
    float
        Difference between the current position and the threshold.
    """
    return float(y[0] - threshold)


# Detect only downward crossings of the threshold.
lower_position_event.direction = -1


def reset_at_event(
    t: float,
    y: NDArray[np.float64],
    reset_position: float,
) -> list[float]:
    """
    Reset the position and place the oscillator in the held mode.

    Parameters
    ----------
    t : float
        Current time. Not used by this transition.
    y : np.ndarray
        State immediately before the event. Not used by this transition.
    reset_position : float
        Position after the reset.

    Returns
    -------
    list of float
        One-dimensional state containing the reset position.
    """
    return [reset_position]


def oscillator_energy(
    values: list[NDArray[np.float64]],
    mass: float,
    stiffness: float,
) -> NDArray[np.float64]:
    """
    Compute oscillator energy at each evaluation time.

    In the held mode, velocity is treated as zero. The result is the sum
    of spring potential energy and kinetic energy.

    Parameters
    ----------
    values : list of np.ndarray
        States evaluated at the statistics collector's evaluation times.
    mass : float
        Oscillator mass.
    stiffness : float
        Spring stiffness.

    Returns
    -------
    np.ndarray
        Energy at each evaluation time.
    """
    return np.array(
        [
            0.5 * stiffness * state[0] ** 2
            + (0.5 * mass * state[1] ** 2 if len(state) == 2 else 0.0)
            for state in values
        ],
        dtype=float,
    )


def position_values(
    values: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Extract positions from states at the evaluation times.

    Parameters
    ----------
    values : list of np.ndarray
        States evaluated at the statistics collector's evaluation times.

    Returns
    -------
    np.ndarray
        Position at each evaluation time.
    """
    return np.array([state[0] for state in values], dtype=float)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    filename: str,
) -> None:
    """
    Apply common plot formatting and optionally save a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to format and optionally save.
    filename : str
        Output filename without extension. The globally configured
        ``fileformat`` is appended.
    """
    for ax in fig.axes:
        ax.xaxis.label.set_size(fontsize_text)
        ax.yaxis.label.set_size(fontsize_text)
        ax.title.set_size(fontsize_text)
        ax.tick_params(axis="both", labelsize=fontsize_ticks)
        ax.grid(True, alpha=alpha_grid)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(fontsize_ticks)
            legend.get_title().set_fontsize(fontsize_ticks)

    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(fontsize_ticks)
        legend.get_title().set_fontsize(fontsize_ticks)

    fig.tight_layout()

    if save_plots:
        plot_folder.mkdir(parents=True, exist_ok=True)
        output_path = plot_folder / f"{filename}.{fileformat}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def plot_trajectory(
    trajectory: FunctionOfVariableDimension,
    title: str,
    filename: str,
) -> None:
    """
    Plot the position and velocity of a trajectory.

    Velocity is displayed as NaN in the held mode because it is not part
    of the state. Dashed vertical lines indicate dimension-changing jumps.

    Parameters
    ----------
    trajectory : FunctionOfVariableDimension
        Trajectory to plot.
    title : str
        Title of the figure.
    filename : str
        Output filename without extension.
    """
    times = np.asarray(trajectory.times)
    positions = [state[0] for state in trajectory.values]
    velocities = [
        state[1] if len(state) == 2 else np.nan for state in trajectory.values
    ]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(times, positions, label="Position x")
    axes[1].plot(times, velocities, label="Velocity v")

    jump_times = trajectory.dimension_jump_times()

    for ax in axes:
        for jump_time in jump_times:
            ax.axvline(
                jump_time,
                color="tab:red",
                linestyle="--",
                alpha=0.4,
            )
        ax.legend()

    axes[0].set_title(title)
    axes[0].set_ylabel("Position")
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel(r"Time $t$")

    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Run the deterministic and stochastic oscillator examples.

    The function optionally saves trajectory and statistics plots in the
    configured output directory and displays the figures.
    """
    # Physical parameters and simulation interval.
    mass = 2.0
    stiffness = 2.0
    damping = 0.0
    flow_args = (mass, stiffness, damping)

    initial_state = [1.0, 0.0]
    t_span = (0.0, 20.0)
    t_eval = np.linspace(*t_span, 401)

    release_factor = -0.3
    forced_jump_times = [5.0, 12.0]

    # -----------------------------------------------------------------------
    # 1. Deterministic hybrid oscillator
    # -----------------------------------------------------------------------
    # A downward crossing of -0.5 resets the position to 0.5 and places the
    # oscillator in the held mode. Prescribed jumps toggle the mode.
    deterministic = solve_hsvd(
        flow_function=oscillator_flow,
        initial_value=initial_state,
        t_span=t_span,
        jump_times=forced_jump_times,
        jump_time_transition_function=toggle_oscillator,
        jump_time_transition_args=(release_factor,),
        jump_set_function=lower_position_event,
        jump_set_args=(-0.5,),
        jump_set_transition_function=reset_at_event,
        jump_set_transition_args=(0.5,),
        flow_args=flow_args,
        t_eval=t_eval,
    )

    print("Deterministic jump times:", deterministic.jump_times())

    plot_trajectory(
        deterministic,
        "Deterministic hybrid oscillator",
        "deterministic",
    )

    # -----------------------------------------------------------------------
    # 2. Single stochastic trajectory
    # -----------------------------------------------------------------------
    # The jump rate is capped at 1, so the supplied bound is valid.
    # A seeded NumPy generator makes the trajectory reproducible.
    stochastic = solve_shsvd_single_run(
        flow_function=oscillator_flow,
        jump_rate_function=jump_rate,
        transition_function=toggle_oscillator,
        initial_value=initial_state,
        t_span=t_span,
        jump_rate_bound=1.0,
        flow_args=flow_args,
        transition_args=(release_factor,),
        jump_times=forced_jump_times,
        jump_time_transition_function=toggle_oscillator,
        jump_time_transition_args=(release_factor,),
        t_eval=t_eval,
        rng=np.random.default_rng(42),
    )

    print("Stochastic jump times:", stochastic.jump_times())
    print("Final state:", stochastic(t_span[1]))

    plot_trajectory(
        stochastic,
        "Single stochastic trajectory",
        "stochastic",
    )

    # -----------------------------------------------------------------------
    # 3. Statistics across multiple trajectories
    # -----------------------------------------------------------------------
    # Collect energy and position as custom evaluations. cef_args contains
    # one argument tuple or None for each custom evaluation function.
    stats = solve_shsvd(
        flow_function=oscillator_flow,
        jump_rate_function=jump_rate,
        transition_function=toggle_oscillator,
        initial_value=initial_state,
        t_span=t_span,
        jump_rate_bound=1.0,
        flow_args=flow_args,
        transition_args=(release_factor,),
        jump_times=forced_jump_times,
        jump_time_transition_function=toggle_oscillator,
        jump_time_transition_args=(release_factor,),
        t_eval=t_eval,
        norm_order=[1, 2],
        collect_dimension=True,
        collect_jump_times=True,
        collect_dimension_jump_times=True,
        custom_evaluation_functions=[oscillator_energy, position_values],
        cef_args=((mass, stiffness), None),
        iterations=100,
        rng=np.random.default_rng(123),
        verbose=True,
        verbose_step=25,
    )

    print("Number of simulated trajectories:", stats.evaluated_functions)
    print("Mean final position:", stats.cef_means[1, -1])
    print("Mean final energy:", stats.cef_means[0, -1])
    print("Mean final dimension:", stats.dimension_means[-1])

    # Shaded bands show ±1 standard deviation across trajectories.
    # They are not confidence intervals for the mean.
    save_figure(
        stats.plot_norm(order=[1, 2], sigma=1),
        "mean_norms",
    )
    save_figure(
        stats.plot_dimension(sigma=1),
        "mean_dimension",
    )
    save_figure(
        stats.plot_cefs(
            sigma=1,
            labels=["Energy", "Position"],
            title="Mean energy and position",
            legend=True,
        ),
        "mean_energy_and_position",
    )
    save_figure(
        stats.plot_number_of_jumps(
            title="Number of jumps per trajectory",
            xlabel="Number of jumps",
            ylabel="Frequency",
        ),
        "jump_counts",
    )

    # -----------------------------------------------------------------------
    # 4. Statistics with stochastic jumps only
    # -----------------------------------------------------------------------
    # No prescribed jump times or jump-set events are supplied.
    stochastic_only_stats = solve_shsvd(
        flow_function=oscillator_flow,
        jump_rate_function=jump_rate,
        transition_function=toggle_oscillator,
        initial_value=initial_state,
        t_span=t_span,
        jump_rate_bound=1.0,
        flow_args=flow_args,
        transition_args=(release_factor,),
        t_eval=t_eval,
        norm_order=[1, 2],
        collect_dimension=True,
        collect_jump_times=True,
        collect_dimension_jump_times=True,
        custom_evaluation_functions=[oscillator_energy, position_values],
        cef_args=((mass, stiffness), None),
        iterations=100,
        rng=np.random.default_rng(456),
        verbose=True,
        verbose_step=25,
    )

    print(
        "Number of stochastic-only trajectories:",
        stochastic_only_stats.evaluated_functions,
    )
    print("Mean final position:", stochastic_only_stats.cef_means[1, -1])
    print("Mean final energy:", stochastic_only_stats.cef_means[0, -1])
    print("Mean final dimension:", stochastic_only_stats.dimension_means[-1])

    save_figure(
        stochastic_only_stats.plot_norm(order=[1, 2], sigma=1),
        "stochastic_only_mean_norms",
    )
    save_figure(
        stochastic_only_stats.plot_dimension(sigma=1),
        "stochastic_only_mean_dimension",
    )
    save_figure(
        stochastic_only_stats.plot_cefs(
            sigma=1,
            labels=["Energy", "Position"],
            title="Mean energy and position: stochastic jumps only",
            legend=True,
        ),
        "stochastic_only_mean_energy_and_position",
    )
    save_figure(
        stochastic_only_stats.plot_number_of_jumps(
            title="Number of stochastic jumps per trajectory",
            xlabel="Number of jumps",
            ylabel="Frequency",
        ),
        "stochastic_only_jump_counts",
    )

    if save_plots:
        print(f"Figures saved to: {plot_folder.resolve()}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
