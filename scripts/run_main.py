import matplotlib.pyplot as plt
import numpy as np

from function_of_variable_dimension import FunctionOfVariableDimension
from stats_function_of_variable_dimension import (
    StatsFunctionsOfVariableDimension,
)

if __name__ == "__main__":
    times = [0, 1, 1, 2, 2, 4, 4, 5]
    values = [[0], [1], [0.5, 0.5], [1.5, 2.5], [4], [6], [8], [9]]
    func = FunctionOfVariableDimension(times, values)
    jt, index = func.jump_times(return_index=True)
    djt = func.dimension_jump_times()
    ndjt = func.non_dimension_jump_times()
    print(jt, index)
    print(djt)
    print(ndjt)
    # func.delete_values(4)
    y = func(1)
    print(y)
    y = func.left_limit(1)
    print(y)
    func.plot_coordinates()
    norm = func.norm()
    print(norm)
    print(func.dimension(1.5))
    func.plot_norm(order=1)
    t_values = np.arange(0, 1.01, 0.05)
    x_values = t_values**2
    func.add_values(t_values, x_values)
    func.delete_values(4, [6])
    func.plot_coordinates()
    func = -func
    func.plot_coordinates()
    func = func + 10
    func.plot_coordinates()
    func = func * 2
    func.plot_coordinates()
    func = 1 / func
    func.plot_coordinates()
    times = [0, 1, 1, 1, 2, 2, 4]
    values = [[0], [1], [1], [0.5, 0.5], [1.5, 2.5], [4], [6]]
    func = FunctionOfVariableDimension(times, values)
    func_2 = FunctionOfVariableDimension(2, 3)

    times = [0, 1, 1, 2, 2, 4]
    values = [[0], [2], [1, 1], [3, 5], [8], [12]]
    times_2 = [4, 5, 6]
    values_2 = [[2, 2, 2], [4, 3, 1], [6, 4, 0]]

    func_new = FunctionOfVariableDimension(times, values)
    func_2 = FunctionOfVariableDimension(times_2, values_2)

    func_conc = func.concatenate(func_2)

    func_test = func + func_new

    func.plot_coordinates()
    func_new.plot_coordinates()
    func_test.plot_coordinates()
    func_2.plot_coordinates()
    func_conc.plot_coordinates()

    test = func([1.5, 3])
    print(test)

    stats = StatsFunctionsOfVariableDimension(
        (0, 4), 0.25, order=[1, 2], collect_jump_times=True
    )
    stats.add_function(func)
    stats.add_function(func_new)
    print(stats.times)
    print(stats.norm_means)
    print(stats.norm_variances)
    stats.plot_norm([1, 2], legend=True)
    stats.plot_dimension(legend=True)
    stats.plot_jump_times(number=[1, 2], legend=True)

    times = [0, 1, 1, 2, 2, 3]
    values = [
        [1 - 0],
        [1 - 0.666],
        [1 - 0.666, 1 - 0],
        [1 - 1, 1 - 0.333],
        [1 - 0.333],
        [1 - 1],
    ]

    func = FunctionOfVariableDimension(times, values)
    fig = func.plot_coordinates(show_jump_times=True)
    plt.figure(fig)
    plt.axhline(0, color="black")
    plt.axhline(1, color="black")
    plt.xlim(0, 2.5)
    plt.savefig("cadlag_function.png")
    plt.show()
    print(func.left_limit([1, 2]))
    func2 = FunctionOfVariableDimension(times, values)
    print(func == func2)
    func3 = func2 + 1
    print(func3 == func)
