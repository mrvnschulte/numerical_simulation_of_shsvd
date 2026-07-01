"""
Provides utility functions to create initial conditions for multi-variant model
"""

import numpy as np
from numpy.typing import ArrayLike

from .model_transformations import map_I_to_x, susceptibles_per_variant
from .utils import variant_in_set


def create_initial_conditions_from_seroprevalence(
    fraction_with_antibodies: float,
    protection_against_variant: ArrayLike,
    infectious_with_variant: ArrayLike,
) -> np.ndarray:
    n_variants = len(protection_against_variant)
    if not n_variants == len(infectious_with_variant):
        raise ValueError(
            "protection_against_variant and infectious_with_variant must be of same length"
        )
    I = create_protection_state_from_seroprevalence(
        fraction_with_antibodies, protection_against_variant, matrix_form=True
    )
    I = distribute_infectious_fraction(I, infectious_with_variant)
    x = map_I_to_x(I)

    return x


def create_protection_state_from_seroprevalence(
    fraction_with_antibodies: float,
    protection_against_variant: ArrayLike,
    matrix_form=True,
) -> np.ndarray:
    n_variants = len(protection_against_variant)
    I = np.zeros((2**n_variants, 2**n_variants))

    I[0, 0] = 1 - fraction_with_antibodies
    for j in range(2**n_variants):
        variants_of_set = [
            variant_in_set(k, j) for k in range(1, n_variants + 1)
        ]
        factors = np.where(
            variants_of_set,
            protection_against_variant,
            1 - np.asarray(protection_against_variant),
        )
        I[0, j] += fraction_with_antibodies * np.prod(factors)

    if matrix_form:
        x = I
    else:
        x = map_I_to_x(I)

    return x


def distribute_infectious_fraction(
    I: np.ndarray, infectious_with_variant: ArrayLike
) -> np.ndarray:
    protected_against_variant = 1 - susceptibles_per_variant(
        I, matrix_form=True
    )
    n_variants = len(infectious_with_variant)
    for k in range(n_variants):
        protection_sets = [
            variant_in_set(k + 1, i) for i in range(2**n_variants)
        ]
        infectious_fraction = (
            infectious_with_variant[k] / protected_against_variant[k]
        )
        I[2**k : 2 ** (k + 1), protection_sets] = (
            infectious_fraction * I[: 2**k, protection_sets]
        )  # TODO : Adjust for
        I[: 2**k, protection_sets] = (1 - infectious_fraction) * I[
            : 2**k, protection_sets
        ]

    return I
