"""
This module provides solvers for (stochastic) hybrid systems of variable
dimension and functionalities to store, compute and plot (statistics of)
functions of variable dimension
"""

from .solve_shsvd import solve_hsvd, solve_shsvd_single_run, solve_shsvd
from .function_of_variable_dimension import FunctionOfVariableDimension
from .stats_functions_of_variable_dimension import (
    StatsFunctionsOfVariableDimension,
)

__all__ = ["FunctionOfVariableDimension", "StatsFunctionsOfVariableDimension"]
