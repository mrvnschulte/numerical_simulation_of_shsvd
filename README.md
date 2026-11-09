# Numerical Simulation of SHSVD

Numerical tools for simulating and analysing **stochastic hybrid systems of variable dimension**.

## Overview

This package provides solvers and analysis tools for systems whose state
dimension may change at jumps. It includes:

- A solver for deterministic hybrid systems with prescribed jumps and jump-set events
- Simulation of individual trajectories of stochastic hybrid systems of variable dimension
- Statistical analysis across multiple stochastic trajectories
- A representation of functions with variable dimension, including jumps and interpolation
- Tools for collecting and plotting statistics of norms, dimensions, jump times, and custom evaluations

## Installation

From the project root:

```bash
python -m pip install .
```

For development, including the test dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.5

Development dependencies:

- pytest >= 7
- pytest-cov

## Model Description

A hybrid system combines continuous dynamics with discrete transitions:

- **Flow:** Between jumps, the state evolves according to a user-defined differential equation.
- **Prescribed jumps:** Transitions occur at specified times.
- **Jump-set events:** Transitions occur when a user-defined event function reaches zero.
- **Stochastic jumps:** Transitions occur randomly with a jump rate which depends on the state of the system.

A transition can change both the state value and its dimension. For example, an
oscillator may switch between an oscillating state `[position, velocity]` and a
held state `[position]`.

Continuous dynamics are integrated with `scipy.integrate.solve_ivp`. The
resulting trajectories are represented as `FunctionOfVariableDimension`
objects.

### Jump Representation

A jump is represented by two consecutive entries with the same time:

1. The first entry stores the **left limit**.
2. The second entry stores the **value after the jump**.

Between stored points, values are interpolated linearly. Evaluating a trajectory
at a jump time returns the value after the jump.

## Project Structure

```text
numerical_simulation_of_shsvd/
├── __init__.py
├── _utils.py
├── function_of_variable_dimension.py
├── solve_hsvd.py
├── solve_shsvd.py
├── solve_shsvd_single_run.py
└── stats_functions_of_variable_dimension.py

examples/
├── oscillator/
│   ├── example_oscillator.py
│   └── stability_deterministic_held_pushed_oscillator.py
├── exponential_decay/
│   └── exponential_decay.py
└── variable_dimension_function/
    └── variable_dimension_function.py

tests/
├── test_function_of_variable_dimension.py
├── test_init.py
├── test_solve_hsvd.py
├── test_solve_shsvd.py
├── test_solve_shsvd_single_run.py
├── test_stats_functions_of_variable_dimension.py
└── test_utils.py

pyproject.toml
README.md
```
````

## Usage

### Deterministic Hybrid System

The following example integrates exponential decay and adds a second coordinate
at a prescribed jump time:

```python
import numpy as np

from numerical_simulation_of_shsvd import solve_hsvd

def flow(t, y):
    return -np.asarray(y)

def add_coordinate(t, y):
    return [y[0], 1.0]

trajectory = solve_hsvd(
    flow_function=flow,
    initial_value=[1.0],
    t_span=(0.0, 2.0),
    jump_times=[1.0],
    jump_time_transition_function=add_coordinate,
    t_eval=[0.0, 0.5, 1.0, 1.5, 2.0],
)

print("Jump times:", trajectory.jump_times())
print("Before the jump:", trajectory.left_limit(1.0))
print("After the jump:", trajectory(1.0))
print("Final value:", trajectory(2.0))
```

A prescribed jump at the initial time is applied before integration.
Prescribed jumps at the final time are not applied. When supplying
`jump_times`, use a nonempty list or NumPy array.

### Single Stochastic Trajectory

The following system has a constant jump rate. Each jump adds a new coordinate:

```python
import numpy as np

from numerical_simulation_of_shsvd import solve_shsvd_single_run

def flow(t, y):
    return -np.asarray(y)

def jump_rate(t, y):
    return 1.0

def transition(t, y):
    return list(y) + [1.0]

trajectory = solve_shsvd_single_run(
    flow_function=flow,
    jump_rate_function=jump_rate,
    transition_function=transition,
    initial_value=[1.0],
    t_span=(0.0, 5.0),
    jump_rate_bound=1.0,
    rng=np.random.default_rng(42),
)

print("Jump times:", trajectory.jump_times())
print("Final state:", trajectory(5.0))
```

Use a valid upper bound for the jump rate over the simulated trajectory.
`check_jump_rate=True` checks the rate at stored values of each computed
segment; it does **not** establish a global bound between those values. If the
bound is `None` or `0`, the solver initializes it from the initial rate and
enables this check.

### Statistics Across Trajectories

Using the model functions defined above:

```python
import numpy as np

from numerical_simulation_of_shsvd import solve_shsvd

stats = solve_shsvd(
    flow_function=flow,
    jump_rate_function=jump_rate,
    transition_function=transition,
    initial_value=[1.0],
    t_span=(0.0, 5.0),
    jump_rate_bound=1.0,
    stepsize=0.1,
    norm_order=[1, 2],
    collect_dimension=True,
    collect_jump_times=True,
    iterations=100,
    rng=np.random.default_rng(42),
)

print("Number of trajectories:", stats.evaluated_functions)
print("Mean norms at t=5:", stats.evaluate_norm(5.0))
print("Mean dimension at t=5:", stats.evaluate_dimension(5.0))

fig = stats.plot_norm(order=[1, 2], sigma=1, grid=True)
```

When `convergence_estimate` is provided, the solver can instead stop based on
changes between consecutive estimated means. This is **not** a
confidence-interval or sampling-error criterion.

The shaded bands in statistics plots represent a specified number of standard
deviations, not confidence intervals.

### Functions of Variable Dimension

```python
from numerical_simulation_of_shsvd import FunctionOfVariableDimension

func = FunctionOfVariableDimension(
    times=[0.0, 1.0, 1.0, 2.0],
    values=[[1.0], [2.0], [2.0, 3.0], [4.0, 5.0]],
)

print("Value at t=0.5:", func(0.5))
print("Left limit at t=1:", func.left_limit(1.0))
print("Value after jump at t=1:", func(1.0))
print("Dimension jump times:", func.dimension_jump_times())
print("2-norm at t=2:", func.norm(t=2.0, order=2))

scaled = 2.0 * func
```

The class also supports adding and deleting stored values, concatenation,
arithmetic, comparisons, and plotting.

### Statistics of Existing Functions

```python
from numerical_simulation_of_shsvd import (
    FunctionOfVariableDimension,
    StatsFunctionsOfVariableDimension,
)

first = FunctionOfVariableDimension(
    [0.0, 1.0, 2.0],
    [[1.0], [2.0], [3.0]],
)
second = FunctionOfVariableDimension(
    [0.0, 1.0, 2.0],
    [[2.0], [4.0], [6.0]],
)

stats = StatsFunctionsOfVariableDimension(
    t_span=(0.0, 2.0),
    stepsize=0.5,
    order=[1, 2],
    collect_dimension=True,
)

stats.add_function(first)
stats.add_function(second)

print("Mean norms at t=1:", stats.evaluate_norm(1.0))
print("Mean dimension at t=1:", stats.evaluate_dimension(1.0))
```

Custom evaluation functions can be registered with
`add_custom_evaluation_function` **before** adding any functions. Compatible
statistics collectors can also be added or subtracted. The package cannot
verify that custom evaluation functions are identical or that the functions
removed by subtraction form a true subset; these conditions must be checked by
the user.

## Example Scripts

| Script | Description |
|--------|-------------|
| `examples/oscillator/example_oscillator.py` | Deterministic and stochastic oscillator trajectories and statistics |
| `examples/oscillator/stability_deterministic_held_pushed_oscillator.py` | Analytical stability-region plot with numerical validation using the hybrid solver |
| `examples/exponential_decay/exponential_decay.py` | Coordinate-dependent exponential decay with stochastic dimension changes |
| `examples/variable_dimension_function/variable_dimension_function.py` | Construction, operations, plotting, and statistics for functions of variable dimension |

Run an example from the project root after installation:

```bash
python examples/oscillator/example_oscillator.py
```

The scripts expose plot settings such as output directory, file format, DPI,
font sizes, and grid transparency. Output paths are relative to the working
directory.

## API Reference

### Solvers

| Function | Description |
|----------|-------------|
| `solve_hsvd` | Solve a deterministic hybrid system with prescribed jumps and jump-set events |
| `solve_shsvd_single_run` | Simulate one stochastic trajectory, optionally with prescribed jumps and jump-set events |
| `solve_shsvd` | Simulate multiple trajectories and collect statistics |

### Function of Variable Dimension

| Class or method | Description |
|-----------------|-------------|
| `FunctionOfVariableDimension` | Represent time-value pairs with potentially changing state dimension |
| `__call__` | Evaluate the function using linear interpolation |
| `left_limit` | Evaluate the left limit at a jump |
| `add_values`, `delete_values` | Modify stored time-value pairs |
| `dimension` | Determine the state dimension |
| `jump_times` | Return stored jump times |
| `dimension_jump_times` | Return jumps that change the state dimension |
| `non_dimension_jump_times` | Return jumps without a dimension change |
| `norm` | Compute norms of selected values |
| `concatenate` | Concatenate compatible functions |
| `plot_coordinates`, `plot_norm` | Plot coordinates or norms |

### Statistics

| Class or method | Description |
|-----------------|-------------|
| `StatsFunctionsOfVariableDimension` | Collect statistics across functions on a common time grid |
| `add_function` | Add a function and update means and sample variances |
| `add_custom_evaluation_function` | Register a custom evaluation before adding functions |
| `evaluate_norm`, `evaluate_norm_variance` | Interpolate stored norm statistics |
| `evaluate_dimension`, `evaluate_dimension_variances` | Interpolate stored dimension statistics |
| `evaluate_cefs`, `evaluate_cef_variances` | Interpolate custom evaluation statistics |
| `plot_norm`, `plot_dimension`, `plot_cefs` | Plot means with standard-deviation bands |
| `plot_jump_times`, `plot_inter_jump_times` | Plot collected jump or inter-jump times |
| `plot_number_of_jumps` | Plot the number of jumps per function |

## Tests

From the project root, after installing the development dependencies:

```bash
python -m pytest
```

The project's pytest configuration enables coverage reporting for
`numerical_simulation_of_shsvd`.

## Documentation

The public classes and functions contain NumPy-style docstrings. For detailed
parameter descriptions, return values, and implementation notes, see the
corresponding module docstrings.

## Citation

A citation will be added when available.

## License

A license has not yet been specified.

## Author

Marvin Schulte —
[marvin.schulte@itwm.fraunhofer.de](mailto:marvin.schulte@itwm.fraunhofer.de)