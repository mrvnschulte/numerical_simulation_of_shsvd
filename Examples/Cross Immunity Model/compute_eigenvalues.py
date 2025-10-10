# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 13:52:35 2025

@author: schulte
"""

from sympy import *

sigma = symbols('sigma')
r01 = symbols('r01')
r02 = symbols('r02')

A = Matrix([[(1-sigma)*r01, 0, (1-sigma)*r01, 0],
            [0, (1-sigma)*r02, 0, (1-sigma)*r02],
            [sigma*r01, 0, sigma*r01, 0],
            [0, sigma*r02, 0, sigma*r02]])

