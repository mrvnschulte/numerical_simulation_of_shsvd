# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:42:37 2025

@author: schulte
"""

import numpy as np
import bisect
import matplotlib.pyplot as plt
import scipy.linalg as la

class function_of_variable_dimension():
    
    def __init__(self, times=None, values=None):
        
        if times is None and values is None:
            
            self.times = []
            self.values = []
            
        else:
            
            times = self.convert_to_list(times)
            values = self.convert_to_list(values)
            
            # Check if initialisation is meaningful
            if not (isinstance(times, (list, np.ndarray)) and isinstance(values, (list, np.ndarray))):
                raise ValueError('Times and/or values are not array-like')
            if not (len(times) == len(values)):
                raise ValueError('Times and values are not of same length')
            for index, value in enumerate(values):
                values[index] = self.convert_to_list(value)
                if (isinstance(values[index], (list, np.ndarray))):
                    continue
                else:
                    raise ValueError(f'{value} in values is not array-like nor can it be converted')
            
            times, values = self.delete_double_values(times, values, initialisation=True)
            
            self.times = times
            self.values = values
            
            self.sort()
            self.check_multiple_jumps()
            self.check_dimension()
            
        
    def __call__(self, t):
        
        if isinstance(t, (list, np.ndarray)):
            value = []
            for time in t:
                value.append(self.interpolate_value(time))
        else:
            value = self.interpolate_value(t)
        
        return value

    
    def convert_to_list(self, number):
        
        if (isinstance(number, (int, float, np.float16, np.float32, np.float64))):
            number = [number]
        elif (isinstance(number, np.ndarray)):
            number = list(number)
            
        return number
        
        
    def sort(self):
        
        sorted_pairs = sorted(zip(self.times, self.values, range(len(self.times))), key=lambda x: (x[0], x[2]))
        times, values = map(list, zip(*[(x[0], x[1]) for x in sorted_pairs]))
        
        self.times = times
        self.values = values
        
        
    def delete_double_values(self, times, values, initialisation=False):
        
        indices_to_remove = []
        
        for index_1, (time, value) in enumerate(zip(times, values)):
            if initialisation:
                indices = [index for index, self_time in enumerate(times) if self_time == time and index > index_1]
            else:
                indices = [index for index, self_time in enumerate(self.times) if self_time == time]
            for index in indices:
                if initialisation:
                    if len(values[index]) == len(value):
                        if np.allclose(values[index], value):
                            indices_to_remove.insert(0, index_1)
                else:
                    if len(self.values[index]) == len(value):
                        if np.allclose(self.values[index], value):
                            indices_to_remove.insert(0, index_1)
                        
        for index in indices_to_remove:
            #print(f'Removed {times[index]}, {values[index]}')
            del times[index]
            del values[index]
            
        return times, values
        
        
    def check_multiple_jumps(self):
        
        unique_times, indices, counts = np.unique(self.times, return_counts=True, return_index=True)
        jump_indices = indices[counts > 2]
        
        if len(jump_indices)>0:
            raise ValueError(f'More than one jump occurs at time(s): {np.array(self.times)[jump_indices]}')
    
        
    def check_dimension(self):
        
        jump_times, jump_indices = self.jump_times(return_index=True)
        dimension_values = self.dimension()
        
        for i in range(len(jump_indices) - 1):
            
            start = jump_indices[i] + 1
            end = jump_indices[i+1] + 1
            
            dimension = dimension_values[start]
            
            for j in range(start + 1, end):
                if dimension_values[j] != dimension:
                    raise ValueError(f'Function changes dimension at {self.times[j]} without specified jump')
        
        
    def interpolate_value(self, t):
        
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(f't is not between {self.times[0]} and {self.times[-1]} (first and last value of times)')
        # elif t == self.times[0]:
        #     if t in self.jump_times():
        #         return self.values[1]
        #     else:
        #         return self.values[0]
        elif t == self.times[-1]:
            return np.array(self.values[-1])
        
        index_t = bisect.bisect_right(self.times, t)
        
        interp_times = np.array(self.times[index_t-1:index_t+1])
        interp_values = np.array(self.values[index_t-1:index_t+1])
        
        value = interp_values[0] + ((t - interp_times[0]) / (interp_times[1] - interp_times[0])) * (interp_values[1] - interp_values[0])
        
        return value
    
    
    def add_values(self, times, values):
        
        times = self.convert_to_list(times)
        values = self.convert_to_list(values)
        for index, value in enumerate(values):
            values[index] = self.convert_to_list(value)
            
        times, values = self.delete_double_values(times, values)
        
        self.times = self.times + times
        self.values = self.values + values
        
        self.sort()
        self.check_multiple_jumps()
        self.check_dimension()
        
        
    def delete_values(self, times=None, values=None, indices=None):
        
        if times == None and values == None and indices == None:
            del times[-1]
            del values[-1]
            
        elif times in self.times:
            
            if times in self.dimension_jump_times():
                raise ValueError('Cannot delete dimension jump time')
            
            if values == None and indices == None:
                index = self.times.index(times)
                del self.times[index]
                del self.values[index]
                if times in self.jump_times():
                    del self.times[index+1]
                    del self.values[index+1]
                
            elif indices == None and isinstance(values, (list, np.ndarray)):
                index = self.times.index(times)
                if self.values[index] != values:
                    if times not in self.jump_times() or self.values[index+1] != values:
                        raise ValueError(f'Time {times} not found with value {values}')
                    del self.times[index+1]
                    del self.values[index+1]
                else:
                    del self.times[index]
                    del self.values[index]
            
            elif values == None and isinstance(indices, (int, float, np.float16, np.float32, np.float64)):
                if self.times[indices] == times:
                    del self.times[indices]
                    del self.values[indices]
                else:
                    raise ValueError(f'Times at index {indices} does not have value {times}')
                    
            elif isinstance(values, (list, np.ndarray)) and isinstance(indices, (int, float, np.float16, np.float32, np.float64)):
                if self.times[indices] == times and self.values[indices] == values:
                    del self.times[indices]
                    del self.values[indices]
                else:
                    raise ValueError(f'Times or values at index {indices} does not have value {times} or {values}, respectively')
                    
            else:
                raise TypeError('Input types do not match')
                
        elif values in self.values:
            if isinstance(indices, (int, float, np.float16, np.float32, np.float64)):
                if self.values[indices] == values:
                    del self.times[indices]
                    del self.values[indices]
            elif indices == None:
                index = self.values.index(values)
                del self.times[index]
                del self.values[index]
                
        elif isinstance(indices, (int, float, np.float16, np.float32, np.float64)):
            del self.times[indices]
            del self.values[indices]
            
        elif isinstance(times, (list, np.ndarray)):
            if values == None and indices == None:
                for time in times:
                    self.delete_values(time)
            elif indices == None:
                for time, value in zip(times, values):
                    self.delete_values(time, value)
            elif values == None:
                for time, index in zip(times, indices):
                    self.delete_values(time, indices=index)
            else:
                for time, value, index in zip(times, values, index):
                    self.delete_values(time, value, index)
                    
        elif isinstance(values, (list, np.ndarray)):
            if indices == None:
                for value in values:
                    self.delete_values(values=value)
            else:
                for value, index in zip(values, indices):
                    self.delete_values(values=value, indices=index)
        
        elif isinstance(indices, (list, np.ndarray)):
            for index in indices:
                self.delete_values(indices=index)
                
        else:
            raise TypeError('Input format not correct')
    
    
    def maximal_time(self):
        
        if len(self.times) > 0:
            return self.times[-1]
        else:
            return -np.infty
        
    
    def minimal_time(self):
        
        if len(self.times) > 0:
            return self.times[0]
        else:
            return np.infty
        
    
    def jump_times(self, return_index=False):
        
        if return_index:
            unique_times, indices, counts = np.unique(self.times, return_counts=True, return_index=return_index)
            jump_indices = indices[counts == 2]
        else:
            unique_times, counts = np.unique(self.times, return_counts=True, return_index=return_index)
        jump_times = unique_times[counts == 2]
        
        if return_index:
            return jump_times,  jump_indices
        else:
            return jump_times
    
    
    def dimension_jump_times(self, return_index=False):
        
        jump_times, indices = self.jump_times(return_index=True)
        dimension_jump_indices = [index for index in indices if len(self.values[index]) != len(self.values[index+1])]
        times = np.array(self.times)
        dimension_jump_times = times[dimension_jump_indices]
        
        if return_index:
            return dimension_jump_times, dimension_jump_indices
        else:
            return dimension_jump_times
    
    
    def non_dimension_jump_times(self, return_index=False):
        
        jump_times = self.jump_times()
        dimension_jump_times = self.dimension_jump_times()
        non_dimension_jump_times = jump_times[~np.isin(jump_times, dimension_jump_times)]
        
        return non_dimension_jump_times
        
        
    def plot_coordinates(self, title='Plot of the coordinates', legend=True, labels=None, xlabel='Time', ylabel='Value', separate_plots=False, sharex=True, sharey=True, grid=False, show_jump_times=False, show_dimension_jump_times=False):
        
        fig = plt.figure()
        dimension_jump_times, dimension_jump_indices = self.dimension_jump_times(return_index=True)
        dimensions = self.dimension()
        max_dimension = max(dimensions)
        prev_index = 0
        
        if not labels:
            labels=[f'Coordinate {i+1}' for i in range(max_dimension)]
            
        if separate_plots:
            figures = []
            for i in range(max_dimension):
                figures.append(plt.figure())
        
        for index in dimension_jump_indices:
            plt.figure(fig)
            plt.gca().set_prop_cycle(None)
            current_dimension = dimensions[prev_index]
            if current_dimension == max_dimension:
                lines = plt.plot(self.times[prev_index:index+1], self.values[prev_index:index+1])
            else:
                plt.plot(self.times[prev_index:index+1], self.values[prev_index:index+1])
                
            if separate_plots:
                for i in range(current_dimension):
                    plt.figure(figures[i])
                    plt.gca().set_prop_cycle(None)
                    values = [value[i] for value in self.values[prev_index:index+1]]
                    plt.plot(self.times[prev_index:index+1], values)
                    
            prev_index = index + 1
        
        plt.figure(fig)
        plt.gca().set_prop_cycle(None)
        current_dimension = dimensions[prev_index]
        if current_dimension == max_dimension:
            lines = plt.plot(self.times[prev_index:], self.values[prev_index:])
        else:
            plt.plot(self.times[prev_index:], self.values[prev_index:])
        
        if title:
            plt.title(title)
        if legend:
            plt.legend(lines, labels)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()
            
        if show_jump_times:
            jump_times = self.jump_times()
            ylim = plt.ylim()
            vlines = plt.vlines(jump_times, ylim[0], ylim[1], 'red', '--')
            plt.ylim(ylim)
            if legend:
                plt.legend(lines + [vlines], labels + ['Jump times'])
                
        if show_dimension_jump_times:
            jump_times = self.dimension_jump_times()
            ylim = plt.ylim()
            dvlines = plt.vlines(jump_times, ylim[0], ylim[1], 'green', '--')
            plt.ylim(ylim)
            if legend:
                if show_jump_times:
                    plt.legend(lines + [vlines, dvlines], labels + ['Non-dimension jump times', 'Dimension jump times'])
                else:
                    plt.legend(lines + [dvlines], labels + ['Dimension jump times'])
            
        if separate_plots and (sharex or sharey or title or grid or show_jump_times):
            xlim = plt.xlim()
            ylim = plt.ylim()
            for i in range(max_dimension):
                plt.figure(figures[i])
                if title:
                    plt.title(f'{labels[i]} plot')
                if grid:
                    plt.grid()
                if sharex:
                    plt.xlim(xlim)
                if sharey:
                    plt.ylim(ylim)
                if show_jump_times:
                    jump_times = self.jump_times()
                    ylim = plt.ylim()
                    vlines = plt.vlines(jump_times, ylim[0], ylim[1], 'red', '--')
                    plt.ylim(ylim)
                    if legend:
                        plt.legend(lines + [vlines], labels + ['Jump times'])
                if show_dimension_jump_times:
                    jump_times = self.dimension_jump_times()
                    ylim = plt.ylim()
                    dvlines = plt.vlines(jump_times, ylim[0], ylim[1], 'green', '--')
                    plt.ylim(ylim)
                    if legend:
                        if show_jump_times:
                            plt.legend(lines + [vlines, dvlines], labels + ['Non-dimension jump times', 'Dimension jump times'])
                        else:
                            plt.legend(lines + [dvlines], labels + ['Dimension jump times'])
                            
        if separate_plots:
            for i in range(current_dimension):
                plt.figure(figures[i])
                plt.gca().set_prop_cycle(None)
                values = [value[i] for value in self.values[prev_index:]]
                plt.plot(self.times[prev_index:], values)
            
        plt.show()
        
        return fig
    
    
    def plot_norm(self, order=2, title='Plot of the norm', legend=True, labels=None, xlabel='Time', ylabel='Value', grid=False, show_jump_times=False, show_dimension_jump_times=False):
        
        fig = plt.figure()
        
        if not isinstance(order, (list, np.ndarray)):
            order_list = [order]
        else:
            order_list = order
        
        if not labels:
            labels = [f'{order} norm' for order in order_list]
        
        lines = []
        
        for order in order_list:
            value_norms = self.norm(order=order)
            line, = plt.plot(self.times, value_norms)
            lines.append(line)
        
        if title:
            plt.title(title)
        if legend:
            plt.legend(lines, labels)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()
        if show_jump_times:
            jump_times = self.jump_times()
            ylim = plt.ylim()
            vlines = plt.vlines(jump_times, ylim[0], ylim[1], 'red', '--')
            plt.ylim(ylim)
            if legend:
                plt.legend(lines + [vlines], labels + ['Jump times'])
        if show_dimension_jump_times:
            jump_times = self.dimension_jump_times()
            ylim = plt.ylim()
            dvlines = plt.vlines(jump_times, ylim[0], ylim[1], 'green', '--')
            plt.ylim(ylim)
            if legend:
                if show_jump_times:
                    plt.legend(lines + [vlines, dvlines], labels + ['Non-dimension jump times', 'Dimension jump times'])
                else:
                    plt.legend(lines + [dvlines], labels + ['Dimension jump times'])
        
        return fig
    
    
    def norm(self, order=None, return_times=False):
        
        value_norms = [float(la.norm(value, ord=order)) for value in self.values]
        
        if return_times:
            return value_norms, self.times
        
        return value_norms
    
    
    def dimension(self, t=None):
        
        if t is None:
            dimension_values = [len(value) for value in self.values]
            return dimension_values
        elif isinstance(t, (int, float, np.float16, np.float32, np.float64, np.int64)):
            dimension_values = self.dimension()
            if t == self.times[-1]:
                return dimension_values[-1]
            index_t = bisect.bisect_right(self.times, t)
            dimension_t = dimension_values[index_t]
            return dimension_t
        elif isinstance(t, (list, np.ndarray, tuple)):
            dimensions_t = [self.dimension(time) for time in t]
            return dimensions_t
        else:
            raise TypeError(f'{t} is not array-like nor a number')
    
    
    def left_limit(self, t):
        
        if t in self.jump_times():
            value = self.values[self.times.index(t)]
        else:
            value = self(t)
            
        return value
    
    
    def concatenate(self, other):
        
        if not isinstance(other, function_of_variable_dimension):
            raise TypeError(f'{other} is not a function of variable dimension')
        
        if self.times[-1] == other.times[0]:
            func = function_of_variable_dimension(self.times + other.times, self.values + other.values)
        
        elif self.times[-1] < other.times[0]:
            if self.dimension[-1] != other.dimension[0]:
                raise ValueError('Dimensions of function do not coincide (only possible when last time for first function and first time for second function are the same)')
                
            func = function_of_variable_dimension(self.times + other.times, self.values + other.values)
            
        else:
            jump_position = 0
            for index_self, t in enumerate(self.times):
                if t >= other.times[0]:
                    if t in other.times:
                        index_other = other.times.index(t) + jump_position
                        if t in other.jump_times():
                            jump_position = 1 - jump_position
                        if self.values[index_self] != other.values[index_other]:
                            raise ValueError('Functions do not coincide at overlapping times')
                    else:
                        if self.dimension(t) != other.dimension(t):
                            raise ValueError('Functions do not coincide at overlapping times')
            
            func = function_of_variable_dimension(self.times + other.times, self.values + other.values)
            
        return func
    
    
class stats_functions_of_variable_dimension():
    
    def __init__(self, t_span, stepsize, order=2, collect_jump_times=False, collect_dimension_jump_times=False):
        
        t0, tf = t_span
        self.times = np.arange(t0, tf, stepsize)
        
        if not np.isclose(self.times[-1], tf):
            self.times = np.append(self.times, tf)
            
        self.evaluated_functions = 0
            
        if isinstance(order, (list, np.ndarray, tuple)):
            self.order = list(order)
        elif isinstance(order, (int, float, np.float16, np.float32, np.float64)):
            self.order = [order]
        else:
            raise TypeError('Order is not array-like nor a number')
            
        self.norm_means = np.zeros((len(self.order), len(self.times)))
        self.norm_variances = np.zeros((len(self.order), len(self.times)))
        self.norm_m2 = np.zeros((len(self.order), len(self.times)))
        self.dimension_means = np.zeros(len(self.times))
        self.dimension_variances = np.zeros(len(self.times))
        self.dimension_m2 = np.zeros(len(self.times))
        self.cef_means = []
        self.cef_variances = []
        self.cef_m2 = []
        
        self.collect_jump_times = collect_jump_times
        if self.collect_jump_times:
            self.jump_times = []
        self.collect_dimension_jump_times = collect_dimension_jump_times
        if self.collect_dimension_jump_times:
            self.dimension_jump_times = []
            
        self.custom_evaluation_functions = []
            
        
        
    def add_function(self, func):
        
        if not isinstance(func, function_of_variable_dimension):
            raise TypeError(f'{func} is not a function of variable dimension')
            
        if func.times[0] > self.times[0] or func.times[-1] < self.times[-1]:
            raise ValueError(f'{func} has values between {func.times[0]} and {func.times[-1]} but they need to be at least between {self.times[0]} and {self.times[-1]}')
            
        self.evaluated_functions += 1
        
        values = func(self.times)

        for index, order in enumerate(self.order):
            
            norm_values = np.array([np.linalg.norm(value, ord=order) for value in values])
            old_means = self.norm_means[index]
            old_m2 = self.norm_m2[index]
            
            new_means, new_variances, new_m2 = self.update_means_and_variances(norm_values, old_means, old_m2)
            
            self.norm_means[index] = new_means
            self.norm_m2[index] = new_m2
            self.norm_variances[index] = new_variances

        
        dimension_values = np.array(func.dimension(self.times))
        old_means = self.dimension_means
        old_m2 = self.dimension_m2
        
        new_means, new_variances, new_m2 = self.update_means_and_variances(dimension_values, old_means, old_m2)

        self.dimension_means = new_means
        self.dimension_m2 = new_m2
        self.dimension_variances = new_variances
        
        # To-Do: Add evaluation of custom evaluation functions
        for index, evaluation_func in enumerate(self.custom_evaluation_functions):
            
            cef_values = np.array(evaluation_func(values))
            old_means = self.cef_means[index]
            old_m2 = self.cef_m2[index]
            
            new_means, new_variances, new_m2 = self.update_means_and_variances(cef_values, old_means, old_m2)
            
            self.cef_means[index] = new_means
            self.cef_m2[index] = new_m2
            self.cef_variances[index] = new_variances

            
        if self.collect_jump_times:
            self.jump_times.append(func.jump_times())
        if self.collect_dimension_jump_times:
            self.dimension_jump_times.append(func.dimension_jump_times())
        
        
    def update_means_and_variances(self, values, old_means, old_m2):
        
        # Update means and variances according to Welfords online algorithm
        
        delta = values - old_means
        new_means = old_means + delta / self.evaluated_functions
        delta2 = values - new_means
        new_m2 = old_m2 + delta * delta2
        if self.evaluated_functions > 1:
            new_variances = new_m2 / (self.evaluated_functions - 1)
        else:
            new_variances = 0
        
        return new_means, new_variances, new_m2
    
        
    def plot_norm(self, order=2, sigma=2, labels=None, title='Plot of mean norm', legend=True, xlabel='Time', ylabel='Norm', grid=False):
        
        fig = plt.figure()
        
        if not isinstance(order, (list, np.ndarray)):
            order_list = [order]
        else:
            order_list = order
        
        if not labels:
            labels = [f'{order} norm' for order in order_list]
        
        lines = []
        
        for order in order_list:
            index = self.order.index(order)
            line, = plt.plot(self.times, self.norm_means[index])
            plt.fill_between(self.times, self.norm_means[index]-sigma*self.norm_variances[index], self.norm_means[index]+sigma*self.norm_variances[index], alpha=.3)
            lines.append(line)
        
        if title:
            plt.title(title)
        if legend:
            plt.legend(lines, labels)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()
        
        return fig
        
        
    def plot_dimension(self, sigma=2, labels=None, title='Plot of mean dimension', legend=False, xlabel='Time', ylabel='Dimension', grid=False):
        
        fig = plt.figure()
        
        lines = []
        
        labels = ['Dimension']
        
        line, = plt.plot(self.times, self.dimension_means)
        plt.fill_between(self.times, self.dimension_means-sigma*self.dimension_variances, self.dimension_means+sigma*self.dimension_variances, alpha=.3)
        lines.append(line)
        
        if title:
            plt.title(title)
        if legend:
            plt.legend(lines, labels)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()
        
        return fig
        
        
    def plot_jump_times(self, number=None, density=False, separate=True, separate_plots=False, title=None, legend=False, xlabel=None, ylabel=None, grid=False):
        
        if not self.collect_jump_times:
            raise ValueError(f'Jump times have not been in collected in {self}')
            
        fig = plt.figure()
        
        if not number:
            plt.hist(np.concatenate(self.jump_times), density=density)
        elif isinstance(number, (list, np.ndarray, tuple)):
            if separate_plots or separate:
                for index in number:
                    index -= 1
                    jump_times = [item[index] for item in self.jump_times if index < len(self.jump_times)]
                    plt.hist(jump_times, density=density, label=f'Jump {index+1}')
                    if separate_plots:
                        plt.figure()
            else:
                jump_times = []
                for index in number:
                    index -= 1
                    jump_times.append([item[index] for item in self.jump_times if index < len(self.jump_times)])
                plt.hist(jump_times, density=density)
        elif isinstance(number, (int, float, np.float16, np.float32, np.float64)):
            number -= 1
            jump_times = [item[int(number)] for item in self.jump_times if int(number) < len(self.jump_times)]
            plt.hist(jump_times, density=density)
            
        if title:
            plt.title(title)
        if legend:
            plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()
        
        return fig
    
    
    def plot_cefs(self, sigma=2, labels=None, title='Plot of cefs', legend=False, xlabel='Time', ylabel='Value', grid=False):
        
        fig = plt.figure()
        
        lines = []
        
        labels = labels or ['Cef {i+1}' for i in range(len(self.custom_evaluation_functions))]
        
        for index in range(len(self.custom_evaluation_functions)):
            line, = plt.plot(self.times, self.cef_means[index])
            plt.fill_between(self.times, self.cef_means[index]-sigma*self.cef_variances[index], self.cef_means[index]+sigma*self.cef_variances[index], alpha=.3)
            lines.append(line)
        
        if title:
            plt.title(title)
        if legend:
            plt.legend(lines, labels)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if grid:
            plt.grid()
        
        return fig
    
    
    def add_custom_evaluation_function(self, func):
        
        if self.evaluated_functions > 0:
            raise ValueError('Can only add custom evaluation function if no functions have been evaluated yet')
            
        self.custom_evaluation_functions.append(func)
        
        self.cef_means.append(np.zeros(len(self.times)))
        self.cef_variances.append(np.zeros(len(self.times)))
        self.cef_m2.append(np.zeros(len(self.times)))
        
        
    def evaluate_cefs(self, t):
        
        value = self.evaluate(t, self.cef_means)
        
        return value
    
    
    def evaluate_norm(self, t):
        
        value = self.evaluate(t, self.norm_means)
        
        return value
    
    
    def evaluate_dimension(self, t):
        
        value = self.evaluate(t, self.dimension_means)
        
        return value
    
    
    def evaluate_cef_variances(self, t):
        
        value = self.evaluate(t, self.cef_variances)
        
        return value
    
    
    def evaluate_norm_variance(self, t):
        
        value = self.evaluate(t, self.norm_variances)
        
        return value
    
    
    def evaluate_dimension_variances(self, t):
        
        value = self.evaluate(t, self.dimension_variances)
        
        return value
        
    
    def evaluate(self, t, values):
        
        if isinstance(t, (list, np.ndarray)):
            value = []
            for time in t:
                value.append(self.interpolate_value(time, values))
        else:
            value = self.interpolate_value(t, values)
        
        return value


    def interpolate_value(self, t, values):
        
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(f't is not between {self.times[0]} and {self.times[-1]} (first and last value of times)')
        elif t == self.times[-1]:
            return np.array(values[:,-1])
        
        index_t = bisect.bisect_right(self.times, t)
        
        interp_times = np.array(self.times[index_t-1:index_t+1])
        interp_values = np.array(values).T[index_t-1:index_t+1]
        
        value = interp_values[0] + ((t - interp_times[0]) / (interp_times[1] - interp_times[0])) * (interp_values[1] - interp_values[0])
        
        return value

    

if __name__ == '__main__':
    times = [0, 1, 1, 2, 2, 4]
    values = [[0], [1], [.5, .5], [1.5, 2.5], [4], [6]]
    func = function_of_variable_dimension(times, values)
    jt, index = func.jump_times(return_index=True)
    djt = func.dimension_jump_times()
    ndjt = func.non_dimension_jump_times()
    print(jt, index)
    print(djt)
    print(ndjt)
    y = func(1)
    print(y)
    y = func.left_limit(1)
    print(y)
    func.plot_coordinates()
    norm = func.norm()
    print(norm)
    print(func.dimension(1.5))
    func.plot_norm(order=1)
    t_values = np.arange(0, 1.01, .05)
    x_values = t_values ** 2
    func.add_values(t_values, x_values)
    func.delete_values(4, [6])
    func.plot_coordinates()
    times = [0, 1, 1, 1, 2, 2, 4]
    values = [[0], [1], [1], [.5, .5], [1.5, 2.5], [4], [6]]
    func = function_of_variable_dimension(times, values)
    func_2 = function_of_variable_dimension(2, 3)
    
    times = [0, 1, 1, 2, 2, 4]
    values = [[0], [2], [1, 1], [3, 5], [8], [12]]
    times_2 = [4, 5, 6]
    values_2 = [[2, 2, 2], [4, 3, 1], [6, 4, 0]]
    
    func_new = function_of_variable_dimension(times, values)
    func_2 = function_of_variable_dimension(times_2, values_2)
    
    func_conc = func.concatenate(func_2)
    
    func.plot_coordinates()
    func_new.plot_coordinates()
    func_2.plot_coordinates()
    func_conc.plot_coordinates()
    
    test = func([1.5, 3])
    print(test)
    
    stats = stats_functions_of_variable_dimension((0, 4), .25, order=[1, 2], collect_jump_times=True)
    stats.add_function(func)
    stats.add_function(func_new)
    print(stats.times)
    print(stats.norm_means)
    print(stats.norm_variances)
    stats.plot_norm([1, 2], legend=True)
    stats.plot_dimension(legend=True)
    stats.plot_jump_times(number=[1, 2], legend=True)
    
    times = [0, 1, 1, 2, 2, 3]
    values = [[1-0], [1-.666], [1-.666, 1-0], [1-1, 1-.333], [1-.333], [1-1]]
    
    func = function_of_variable_dimension(times, values)
    fig = func.plot_coordinates(show_jump_times=True)
    plt.figure(fig)
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    plt.xlim(0, 2.5)
    plt.savefig('cadlag_function.png')
    plt.show()