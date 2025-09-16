# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:18:05 2025

@author: schulte
"""

from setuptools import setup, find_packages

setup(
    name='numerical_simulation_of_shsvd',
    version='0.1.0',
    author='Marvin Schulte',
    author_email='marvin.schulte@itwm.fraunhofer.de',
    description='A module for simulating stochastic hybrid systems of '
        'variable dimension',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)