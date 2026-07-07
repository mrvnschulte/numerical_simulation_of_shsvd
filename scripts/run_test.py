from numerical_simulation_of_shsvd import FunctionOfVariableDimension

f = FunctionOfVariableDimension([0.0, 1.0, 1.0], [[1], [2], [1, 2]])
n = len(f.times)
f.add_values(0.0, 1)
