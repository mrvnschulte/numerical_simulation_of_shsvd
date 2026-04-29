import numpy as np
from numpy.typing import ArrayLike

from cross_immunity_model import basic_reproduction_number


def find_stable_multi_variant_equilibrium(
    beta: ArrayLike,
    gamma: ArrayLike,
    sigma: ArrayLike,
    b: float,
    atol: float = 1e-3,
):
    r0 = basic_reproduction_number(beta, gamma, b)
    if np.all(r0 < 1):
        return []
    most_transmissible_variant = np.argmax(r0)
    coexisting_variants = [most_transmissible_variant]
