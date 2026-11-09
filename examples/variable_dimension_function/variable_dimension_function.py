"""
Demonstrate functions of variable dimension and their statistics.

The example covers:

1. Constructing and evaluating functions with jumps.
2. Inspecting dimensions, jump times, left limits, and norms.
3. Adding and deleting values, arithmetic, comparison, and concatenation.
4. Collecting statistics, including a custom evaluation function.
5. Combining and subtracting statistics collectors.
6. Plotting trajectories and statistics.

Run from an environment in which numerical_simulation_of_shsvd is installed:

    python example_functions_and_statistics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from numerical_simulation_of_shsvd import (
    FunctionOfVariableDimension,
    StatsFunctionsOfVariableDimension,
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
# Example data
# ---------------------------------------------------------------------------


def make_example_function(
    scale: float = 1.0,
) -> FunctionOfVariableDimension:
    """
    Construct an example function with dimension and value jumps.

    The function changes from one to two dimensions at t=1 and returns
    to one dimension at t=2. At t=3, it has a jump without a dimension
    change.

    Parameters
    ----------
    scale : float, optional
        Factor applied to every coordinate. The default is 1.0.

    Returns
    -------
    FunctionOfVariableDimension
        Function defined on [0, 4].
    """
    times = [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    values = [
        [1.0],
        [2.0],
        [1.0, 1.0],
        [2.0, 3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
    ]
    scaled_values = [
        [scale * coordinate for coordinate in state] for state in values
    ]

    return FunctionOfVariableDimension(times, scaled_values)


def first_coordinate(
    values: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Extract the first coordinate at each evaluation time.

    Parameters
    ----------
    values : list of np.ndarray
        States evaluated at the statistics collector's evaluation times.

    Returns
    -------
    np.ndarray
        First coordinate at each evaluation time.
    """
    return np.array([state[0] for state in values], dtype=float)


def make_collector() -> StatsFunctionsOfVariableDimension:
    """
    Create a statistics collector with a custom evaluation function.

    Returns
    -------
    StatsFunctionsOfVariableDimension
        Empty collector configured for the interval [0, 4].
    """
    stats = StatsFunctionsOfVariableDimension(
        t_span=(0.0, 4.0),
        stepsize=0.25,
        order=[1, 2],
        collect_dimension=True,
        collect_jump_times=True,
        collect_dimension_jump_times=True,
    )
    stats.add_custom_evaluation_function(first_coordinate)
    return stats


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
        Output filename without extension.
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


def plot_coordinates(
    func: FunctionOfVariableDimension,
    filename: str,
) -> None:
    """
    Plot each coordinate of a function of variable dimension.

    Coordinates that do not exist at a stored time are represented by NaN.
    Vertical lines mark dimension-changing and other jumps separately.

    Parameters
    ----------
    func : FunctionOfVariableDimension
        Function to plot.
    filename : str
        Output filename without extension.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_dimension = max(func.dimension())

    for coordinate in range(max_dimension):
        values = [
            state[coordinate] if coordinate < len(state) else np.nan
            for state in func.values
        ]
        ax.plot(
            func.times,
            values,
            marker=".",
            label=f"Coordinate {coordinate + 1}",
        )

    for index, jump_time in enumerate(func.dimension_jump_times()):
        ax.axvline(
            jump_time,
            color="tab:green",
            linestyle="--",
            alpha=0.6,
            label="Dimension jump" if index == 0 else None,
        )

    for index, jump_time in enumerate(func.non_dimension_jump_times()):
        ax.axvline(
            jump_time,
            color="tab:red",
            linestyle=":",
            alpha=0.6,
            label="Value jump" if index == 0 else None,
        )

    ax.set_title("Function of variable dimension")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel("Coordinate value")
    ax.legend()

    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------


def demonstrate_function(func: FunctionOfVariableDimension) -> None:
    """
    Demonstrate evaluation, jumps, dimensions, and norms.

    Parameters
    ----------
    func : FunctionOfVariableDimension
        Example function to inspect.
    """
    print("\n1. Function evaluation and jumps")
    print("Stored times:", func.times)
    print("Stored values:", func.values)

    # Values between stored times are linearly interpolated.
    print("Value at t=0.5:", func(0.5))
    print("Values at t=0.5 and t=1.5:", func([0.5, 1.5]))

    # At a jump, evaluation returns the right-hand value.
    print("Value at jump t=1:", func(1.0))
    print("Left limit at t=1:", func.left_limit(1.0))
    print("Left limits at t=1 and t=2:", func.left_limit([1.0, 2.0]))

    jump_times, jump_indices = func.jump_times(return_index=True)
    print("Jump times:", jump_times)
    print("Indices of left-limit entries:", jump_indices)
    print("Dimension jump times:", func.dimension_jump_times())
    print("Other jump times:", func.non_dimension_jump_times())

    print("Dimensions at stored entries:", func.dimension())
    print("Dimensions at t=0.5, 1.5, 2.5:", func.dimension([0.5, 1.5, 2.5]))
    print("1-norms at t=0.5 and 1.5:", func.norm(t=[0.5, 1.5], order=1))
    print("Time range:", (func.minimal_time(), func.maximal_time()))

    plot_coordinates(func, "function_coordinates")
    save_figure(
        func.plot_norm(
            order=[1, 2],
            title="Norms of the function",
            xlabel=r"Time $t$",
            ylabel="Norm",
            show_dimension_jump_times=True,
        ),
        "function_norms",
    )


def demonstrate_operations(func: FunctionOfVariableDimension) -> None:
    """
    Demonstrate modification, arithmetic, comparison, and concatenation.

    The original function is not modified.

    Parameters
    ----------
    func : FunctionOfVariableDimension
        Example function used as the starting point.
    """
    print("\n2. Modification and operations")

    # Construct an independent copy before modifying stored values.
    modified = FunctionOfVariableDimension(
        list(func.times),
        [list(state) for state in func.values],
    )
    modified.add_values(0.5, [1.5])
    print("After adding t=0.5:", modified.times)
    modified.delete_values(times=0.5)
    print("After deleting t=0.5:", modified.times)

    scaled = 2.0 * func
    shifted = func + 1.0
    combined = func + scaled

    print("Scaled value at t=1.5:", scaled(1.5))
    print("Shifted value at t=1.5:", shifted(1.5))
    print("Sum at t=1.5:", combined(1.5))
    print("Independent copy equals original:", modified == func)

    # The continuation starts at the final value of func.
    continuation = FunctionOfVariableDimension(
        [4.0, 5.0, 6.0],
        [[7.0], [8.0], [9.0]],
    )
    extended = func.concatenate(continuation)
    print(
        "Concatenated time range:",
        (extended.minimal_time(), extended.maximal_time()),
    )
    print("Concatenated value at t=5:", extended(5.0))


def demonstrate_statistics() -> None:
    """
    Demonstrate collecting, evaluating, combining, and plotting statistics.

    Three scaled versions of the same function are evaluated on a common
    time grid. Their statistics are also collected separately to demonstrate
    addition and subtraction of compatible collectors.
    """
    print("\n3. Statistics")

    functions = [make_example_function(scale) for scale in (1.0, 2.0, 3.0)]

    stats = make_collector()
    for func in functions:
        stats.add_function(func)

    print("Number of functions:", stats.evaluated_functions)
    print("Evaluation times:", stats.times)
    print("Stored norm orders:", stats.order)
    print("Mean 1- and 2-norm at t=1.5:", stats.evaluate_norm(1.5))
    print("Norm variances at t=1.5:", stats.evaluate_norm_variance(1.5))
    print("Mean dimension at t=1.5:", stats.evaluate_dimension(1.5))
    print(
        "Dimension variance at t=1.5:",
        stats.evaluate_dimension_variances(1.5),
    )
    print("Mean first coordinate at t=1.5:", stats.evaluate_cefs(1.5))
    print(
        "Variance of first coordinate at t=1.5:",
        stats.evaluate_cef_variances(1.5),
    )
    print("Collected jump times:", stats.jump_times)
    print("Collected dimension jump times:", stats.dimension_jump_times)

    # Collect the first two functions and the third function separately.
    first_part = make_collector()
    first_part.add_function(functions[0])
    first_part.add_function(functions[1])

    second_part = make_collector()
    second_part.add_function(functions[2])

    # The package warns that it cannot verify custom-function identity or
    # the subset relationship required for subtraction.
    combined = first_part + second_part
    remaining = combined - second_part

    print("\nCombined function count:", combined.evaluated_functions)
    print(
        "Combined norm means match direct collection:",
        np.allclose(combined.norm_means, stats.norm_means),
    )
    print(
        "Remaining function count after subtraction:",
        remaining.evaluated_functions,
    )
    print(
        "Remaining norm means match first collector:",
        np.allclose(remaining.norm_means, first_part.norm_means),
    )

    # Shaded bands show ±1 sample standard deviation, not confidence intervals.
    save_figure(
        stats.plot_norm(
            order=[1, 2],
            sigma=1,
            title="Mean norms",
            xlabel=r"Time $t$",
            ylabel="Norm",
        ),
        "statistics_norms",
    )
    save_figure(
        stats.plot_dimension(
            sigma=1,
            title="Mean dimension",
            xlabel=r"Time $t$",
            legend=True,
        ),
        "statistics_dimension",
    )
    save_figure(
        stats.plot_cefs(
            sigma=1,
            labels=["First coordinate"],
            title="Mean first coordinate",
            xlabel=r"Time $t$",
            ylabel="Value",
            legend=True,
        ),
        "statistics_first_coordinate",
    )
    save_figure(
        stats.plot_number_of_jumps(
            title="Number of jumps per function",
            xlabel="Number of jumps",
            ylabel="Frequency",
        ),
        "statistics_jump_counts",
    )


def main() -> None:
    """
    Run the function and statistics demonstrations.

    Figures are optionally saved and displayed according to the plot settings.
    """
    func = make_example_function()
    demonstrate_function(func)
    demonstrate_operations(func)
    demonstrate_statistics()

    if save_plots:
        print(f"\nFigures saved to: {plot_folder.resolve()}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
