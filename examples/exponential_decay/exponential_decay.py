"""
Simulate a stochastic system with coordinate-dependent exponential decay.

The n-th coordinate decays at rate 1/n. At each stochastic jump, the last
coordinate moves to a new coordinate and its previous entry is set to zero.
The dimension therefore increases by one at every jump.

The example computes a sample trajectory and statistics across multiple
trajectories. It illustrates that exponential decay in each fixed dimension
does not by itself imply exponential stability of the stochastic hybrid
system.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from numerical_simulation_of_shsvd import (
    FunctionOfVariableDimension,
    solve_shsvd,
    solve_shsvd_single_run,
)


# ---------------------------------------------------------------------------
# Plot and output settings
# ---------------------------------------------------------------------------

plot_folder = Path("plots")
fileformat = "pdf"
dpi = 1200

fontsize_text = 20
fontsize_ticks = 16
alpha_grid = 0.7

save_plots = True
show_plots = True


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------

initial_value = [1.0]
t_span = (0.0, 5.0)
jump_rate_bound = 1.0
stepsize = 0.01

min_iterations = 50
max_iterations = 1000
tolerance = 1e-4

statistics_seed = 42
trajectory_seed = 43


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def exponential_decay(
    t: float,
    x: NDArray[np.float64] | list[float],
) -> list[float]:
    """
    Compute coordinate-dependent exponential decay.

    The i-th coordinate, using one-based indexing, decays at rate 1/i.

    Parameters
    ----------
    t : float
        Current time. Not used by this autonomous system.
    x : np.ndarray or list of float
        Current state.

    Returns
    -------
    list of float
        Derivatives of the state coordinates.
    """
    return [-float(value) / (index + 1) for index, value in enumerate(x)]


def jump_rate_function(
    t: float,
    x: NDArray[np.float64] | list[float],
) -> float:
    """
    Return the constant stochastic jump rate.

    Parameters
    ----------
    t : float
        Current time. Not used by this rate function.
    x : np.ndarray or list of float
        Current state. Not used by this rate function.

    Returns
    -------
    float
        Constant jump rate of 1.
    """
    return 1.0


def transition_function(
    t: float,
    x: NDArray[np.float64] | list[float],
) -> list[float]:
    """
    Move the last state component to a new coordinate.

    The previous last component is set to zero, and its value is appended
    as a new component. The state dimension increases by one.

    Parameters
    ----------
    t : float
        Current time. Not used by this transition.
    x : np.ndarray or list of float
        State immediately before the jump. Must be nonempty.

    Returns
    -------
    list of float
        State immediately after the jump.

    Raises
    ------
    ValueError
        If the state is empty.
    """
    new_state = list(x)

    if not new_state:
        raise ValueError("The state must not be empty.")

    last_value = float(new_state[-1])
    new_state[-1] = 0.0
    new_state.append(last_value)

    return new_state


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Apply common formatting and optionally save a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to format and optionally save.
    filename : str
        Output filename without extension. The configured file format is
        appended.
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
    filename: str,
) -> None:
    """
    Plot all coordinates of a trajectory.

    Coordinates that do not yet exist are represented by NaN. Vertical
    dashed lines indicate dimension-changing jumps.

    Parameters
    ----------
    trajectory : FunctionOfVariableDimension
        Sample trajectory to plot.
    filename : str
        Output filename without extension.
    """
    times = np.asarray(trajectory.times)
    max_dimension = max(len(state) for state in trajectory.values)

    fig, ax = plt.subplots(figsize=(10, 6))

    for coordinate in range(max_dimension):
        values = [
            state[coordinate] if coordinate < len(state) else np.nan
            for state in trajectory.values
        ]
        ax.plot(times, values, label=f"Coordinate {coordinate + 1}")

    for jump_time in trajectory.dimension_jump_times():
        ax.axvline(
            jump_time,
            color="tab:red",
            linestyle="--",
            alpha=0.4,
        )

    ax.set_title("Coordinates of a sample trajectory")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel("Coordinate value")
    ax.set_xlim(t_span)
    ax.legend()

    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Simulate the system and plot its mean norm and one sample trajectory.

    The number of trajectories is determined by the change in the estimated
    mean norm, subject to the configured minimum and maximum iteration counts.
    Figures are optionally saved and displayed.
    """
    # -----------------------------------------------------------------------
    # 1. Statistics across stochastic trajectories
    # -----------------------------------------------------------------------
    # The convergence criterion measures changes between consecutive mean
    # estimates. It is not a confidence-interval or sampling-error criterion.
    stats = solve_shsvd(
        flow_function=exponential_decay,
        jump_rate_function=jump_rate_function,
        transition_function=transition_function,
        initial_value=initial_value,
        t_span=t_span,
        jump_rate_bound=jump_rate_bound,
        stepsize=stepsize,
        norm_order=1,
        collect_dimension=True,
        collect_jump_times=True,
        convergence_estimate=("norm", 1),
        convergence_estimate_type="max",
        tol=tolerance,
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        rng=np.random.default_rng(statistics_seed),
        verbose=True,
        verbose_step=50,
    )

    print("Number of simulated trajectories:", stats.evaluated_functions)
    print("Mean final 1-norm:", stats.norm_means[0, -1])
    print("Mean final dimension:", stats.dimension_means[-1])

    # The shaded band represents ±1 standard deviation across trajectories,
    # not a confidence interval for the mean.
    norm_figure = stats.plot_norm(
        order=1,
        sigma=1,
        title="Mean distance to the origin",
        xlabel=r"Time $t$",
        ylabel="1-norm",
    )
    norm_figure.axes[0].set_xlim(t_span)
    save_figure(norm_figure, "mean_norm")

    # -----------------------------------------------------------------------
    # 2. Single stochastic trajectory
    # -----------------------------------------------------------------------
    # A separate seeded generator makes this sample path reproducible
    # independently of the number of statistical iterations.
    trajectory = solve_shsvd_single_run(
        flow_function=exponential_decay,
        jump_rate_function=jump_rate_function,
        transition_function=transition_function,
        initial_value=initial_value,
        t_span=t_span,
        jump_rate_bound=jump_rate_bound,
        rng=np.random.default_rng(trajectory_seed),
    )

    print("Sample trajectory jump times:", trajectory.jump_times())
    print("Sample trajectory final state:", trajectory(t_span[1]))

    plot_trajectory(trajectory, "sample_trajectory")

    if save_plots:
        print(f"Figures saved to: {plot_folder.resolve()}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
