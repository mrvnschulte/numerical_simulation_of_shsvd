# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:42:37 2025

@author: schulte
"""

import numpy as np
import bisect
import matplotlib.pyplot as plt
import scipy.linalg as la

class FunctionOfVariableDimension():
    """
    A class for functions of variable dimension.
    
    A class to represent a function of variable dimension defined by 
    time-value pairs.
    
    Attributes:
        times: list
            A list of time points of the function.
        values: list
            A list of corresponding values at the given time points.
    """
    
    def __init__(self, times=None, values=None):
        """
        Initializes the function with given times and values.
        
        Parameters:
            times: float, list or array-like, optional
                A sequence of time points.
            values: float, list or array-like, optional
                A sequence of values corresponding to the times.
        
        Raises:
            ValueError: If times and values are not the same length or not
            array-like.
        """
        if times is None and values is None:
            self.times = []
            self.values = []
            
        else:
            times = self._convert_to_list(times)
            values = self._convert_to_list(values)
            
            # Check if initialisation is meaningful
            if not (isinstance(times, (list, np.ndarray)) and
                    isinstance(values, (list, np.ndarray))):
                raise ValueError('Times and/or values are not array-like')
            if not (len(times) == len(values)):
                raise ValueError('Times and values are not of same length')
            for index, value in enumerate(values):
                values[index] = self._convert_to_list(value)
                if (isinstance(values[index], (list, np.ndarray))):
                    continue
                else:
                    raise ValueError(f'{value} in values is not array-like \
                                     nor can it be converted')
            
            times, values = self._delete_double_values(times, values,
                                                      initialisation=True)
            
            self.times = times
            self.values = values
            
            # Sanity checks
            self._sort()
            self._check_multiple_jumps()
            self._check_dimension()
                
    def __call__(self, t):
        """
        Returns the interpolated value(s) of the function at the given time(s).

        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the function.
        
        Returns:
            value: float or list
                The interpolated value(s) at the given time(s).
        """
        if isinstance(t, (list, np.ndarray)):
            value = []
            for time in t:
                value.append(self._interpolate_value(time))
        else:
            value = self._interpolate_value(t)
        
        return value

    @staticmethod
    def _convert_to_list(number):
        """
        Converts a number or array-like object to a list.
        
        Parameters:
            number: int, float, or np.ndarray
                The input to be converted.
        
        Returns:
            number: list
                Converted list.
        """
        if (isinstance(number, (int, float, np.float16, np.float32,
                                np.float64, np.int16, np.int32, np.int64))):
            number = [number]
        elif (isinstance(number, np.ndarray)):
            number = list(number)
            
        return number
        
    def _sort(self):
        """
        Sorts the times and corresponding values in ascending order of time.
        
        Sorts the times and corresponding values in ascending order of time.
        If there are times which occur twice, then the method preserves the
        original order. This is important for jumps of the function.
        """
        sorted_pairs = sorted(zip(self.times, self.values,
                                  range(len(self.times))),
                              key=lambda x: (x[0], x[2]))
        times, values = map(list, zip(*[(x[0], x[1]) for x in sorted_pairs]))
        
        self.times = times
        self.values = values
        
    def _delete_double_values(self, times, values, initialisation=False):
        """
        Removes duplicate time-value pairs from the lists.
        
        Removes duplicate time-value pairs from times, values during
        initialisation and duplicates between times, values and self.times
        and self.values after initialisation.

        Parameters:
            times: list
                The list of times.
            values: list
                The list of values.
            initialisation: bool, optional
                Indicates if this is during initialization (default is False).
        
        Returns:
            tuple: Updated lists of times and values without duplicates.
        """
        indices_to_remove = []
        
        for index_1, (time, value) in enumerate(zip(times, values)):
            # Find all duplicate times
            if initialisation:
                indices = [index for index, self_time in enumerate(times)
                           if self_time == time and index > index_1]
            else:
                indices = [index for index, self_time in enumerate(self.times)
                           if self_time == time]
            # Check whether found duplicate times are not jumps
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
            del times[index]
            del values[index]
            
        return times, values
        
    def _check_multiple_jumps(self):
        """
        Raises an error if multiple jumps occur at the same time.
        
        Raises:
            ValueError: If more than one jump occurs at the same time.
            
        """
        unique_times, indices, counts = np.unique(self.times,
                                                  return_counts=True,
                                                  return_index=True)
        jump_indices = indices[counts > 2]
        
        if len(jump_indices)>0:
            raise ValueError(f'More than one jump occurs at time(s): \
                             {np.array(self.times)[jump_indices]}')
        
    def _check_dimension(self):
        """
        Checks if the function changes dimension unexpectedly.
        
        Checks if the function changes dimension at a time where no two
        values are given, i.e. the one before and the one after the change of
        dimension.

        Raises:
            ValueError: If the function changes dimension unexpectedly.
        """
        jump_times, jump_indices = self.jump_times(return_index=True)
        dimension_values = self.dimension()
        
        for i in range(len(jump_indices) - 1):
            
            start = jump_indices[i] + 1
            end = jump_indices[i+1] + 1
            
            dimension = dimension_values[start]
            
            for j in range(start + 1, end):
                if dimension_values[j] != dimension:
                    raise ValueError(f'Function changes dimension at \
                                     {self.times[j]} without specified jump')
        
    def _interpolate_value(self, t):
        """
        Interpolates the value of the function at a given time.

        Parameters:
            t: float
                The time at which to interpolate the value.
        
        Returns:
            value: float
                The interpolated value.
        
        Raises:
            ValueError: If the time is outside the range of defined times.
        """
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(f't is not between {self.times[0]} and \
                             {self.times[-1]} (first and last value of times)')
        elif t == self.times[-1]:
            return np.array(self.values[-1])
        
        index_t = bisect.bisect_right(self.times, t)
        
        interp_times = np.array(self.times[index_t-1:index_t+1])
        interp_values = np.array(self.values[index_t-1:index_t+1])
        
        value = interp_values[0] + (((t - interp_times[0]) /
                                    (interp_times[1] - interp_times[0]))
                                    * (interp_values[1] - interp_values[0]))
        
        return value
    
    def add_values(self, times, values):
        """
        Adds new time-value pairs to the function.

        Parameters:
            times: float, list or array-like
                The new time(s) to add.
            values: float, list or array-like
                The new value(s) corresponding to the times.
        """
        # Convert everything to lists
        if isinstance(times, (int, float, np.float16, np.float32, np.float64,
                              np.int16, np.int32, np.int64)):
            single_value = True
        else:
            single_value = False
        times = self._convert_to_list(times)
        values = self._convert_to_list(values)
        if not single_value:
            for index, value in enumerate(values):
                values[index] = self._convert_to_list(value)
        else:
            values = [values]
            
        times, values = self._delete_double_values(times, values)
        
        self.times = self.times + times
        self.values = self.values + values
        
        # Sanity checks
        self._sort()
        self._check_multiple_jumps()
        self._check_dimension()
        
    def delete_values(self, times=None, values=None, indices=None):
        """
        Deletes time-value pairs from the function.
        
        If nothing is provided to the function, it deletes the last time-value
        pair. Otherwise, the time and/or value and/or index of the time-value
        pair to be deleted, can be provided.

        Parameters:
            times: float or list, optional
                The time(s) to delete.
            values: list, optional
                The value(s) to delete.
            indices: int or list, optional
                The indices of the pairs to delete.
                
        Raises:
            ValueError: If a time to be deleted is a dimension jump time or
                a given time-value(-index) pair is not part of the function.
            TypeError: If input types do not match, i.e. one time, but
                multiple values and/or indices are provided.
        """
        # If nothing is provided, delete the last time-value pair
        if times == None and values == None and indices == None:
            del self.times[-1]
            del self.values[-1]
            
        elif times in self.times:
            # Dimension jump times cannot be deleted
            if times in self.dimension_jump_times():
                raise ValueError('Cannot delete dimension jump time')
            
            # If only a time is provided, delete this (twice if jump time)
            if values == None and indices == None:
                index = self.times.index(times)
                if times in self.jump_times():
                    del self.times[index+1]
                    del self.values[index+1]
                del self.times[index]
                del self.values[index]
                
            # If time and value is provided, delete this if existing
            elif indices == None and isinstance(values, (list, np.ndarray)):
                index = self.times.index(times)
                if self.values[index] != values:
                    if (times not in self.jump_times()
                        or self.values[index+1] != values):
                        raise ValueError(f'Time {times} not found with value \
                                         {values}')
                    del self.times[index+1]
                    del self.values[index+1]
                else:
                    del self.times[index]
                    del self.values[index]
            
            # If time and index is provided, delete this if time is at index
            elif values == None and isinstance(indices, (int, np.int16,
                                                         np.int32, np.int64)):
                if self.times[indices] == times:
                    del self.times[indices]
                    del self.values[indices]
                else:
                    raise ValueError(f'Times at index {indices} does not \
                                     have value {times}')
                    
            # If time, value and index are provided, delete this if they match
            elif (isinstance(values, (list, np.ndarray))
                  and isinstance(indices, (int, np.int16, np.int32, 
                                           np.int64))):
                if (self.times[indices] == times
                    and self.values[indices] == values):
                    del self.times[indices]
                    del self.values[indices]
                else:
                    raise ValueError(f'Times or values at index {indices} \
                                     does not have value {times} or {values},\
                                         respectively')
                    
            else:
                raise TypeError('Input types do not match')
            
        # If multiple times are provided, delete all of them together with
        # the corresponding values and/or indices
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
        
        # If time is not stores, raise an error
        elif isinstance(times, (int, float, np.float16, np.float32,
                                np.float64, np.int16, np.int32, np.int64)):
            raise ValueError(f'{times} is not in the times of the function')

        elif values in self.values:
            # If value and index are provided, delete this if they match
            if isinstance(indices, (int, float, np.float16, np.float32,
                                    np.float64, np.int16, np.int32, np.int64)):
                if self.values[indices] == values:
                    del self.times[indices]
                    del self.values[indices]
            # If only value is provided, search for this and delete it
            elif indices == None:
                index = self.values.index(values)
                del self.times[index]
                del self.values[index]
         
        # If multiple values are provided, delete each together with the
        # corresponding index if provided
        elif isinstance(values, (list, np.ndarray)):
            if indices == None:
                for value in values:
                    self.delete_values(values=value)
            else:
                for value, index in zip(values, indices):
                    self.delete_values(values=value, indices=index)
                   
        # If value is provided, that is not stored, raise an error
        elif values is not None:
            raise ValueError('Value is stored as a value of the function')

        # If an index is provided, delete the corresponding time-value pair
        elif isinstance(indices, (int, np.int16, np.int32, np.int64)):
            if indices > len(self.times):
                raise ValueError(f'Index {indices} is greater than number of\
                                 stored time-value pairs ({len(self.times)})')
            del self.times[indices]
            del self.values[indices]
            
        # If multiple indices are provided, delete the corresponding time-value
        # pairs
        elif isinstance(indices, (list, np.ndarray)):
            for index in indices:
                self.delete_values(indices=index)
                                
        else:
            raise TypeError('Input format not correct')
    
    def maximal_time(self):
        """
        Returns the maximum time in the function.

        Returns:
            float: The maximum time, or -inf if no times are defined.
        """
        if len(self.times) > 0:
            return self.times[-1]
        else:
            return -np.infty
    
    def minimal_time(self):
        """
        Returns the minimum time in the function.

        Returns:
            float: The minimum time, or +inf if no times are defined.
        """
        if len(self.times) > 0:
            return self.times[0]
        else:
            return np.infty
    
    def jump_times(self, return_index=False):
        """
        Returns the times at which jumps occur in the function.

        Parameters:
            return_index: bool, optional
                If True, also return the indices of the jump times.
        
        Returns:
            jump_times: list
                The times at which jumps occur.
            indices: list, optional
                The indices of the jump times, if requested.
        """
        if return_index:
            unique_times, indices, counts = np.unique(self.times,
                                                      return_counts=True,
                                                      return_index=
                                                      return_index)
            jump_indices = indices[counts == 2]
        else:
            unique_times, counts = np.unique(self.times, return_counts=True,
                                             return_index=return_index)
        jump_times = unique_times[counts == 2]
        
        if return_index:
            return jump_times,  jump_indices
        else:
            return jump_times
    
    def dimension_jump_times(self, return_index=False):
        """
        Returns the times at which dimension jumps occur.

        Parameters:
            return_index: bool, optional
                If True, also return the indices of the dimension jump times.
        
        Returns:
            dimension_jump_times: list
                The times at which dimension jumps occur.
            indices: list, optional
                The indices of the dimension jump times, if requested.
        """
        jump_times, indices = self.jump_times(return_index=True)
        dimension_jump_indices = [index for index in indices
                                  if (len(self.values[index])
                                      != len(self.values[index+1]))]
        times = np.array(self.times)
        dimension_jump_times = times[dimension_jump_indices]
        
        if return_index:
            return dimension_jump_times, dimension_jump_indices
        else:
            return dimension_jump_times
    
    def non_dimension_jump_times(self, return_index=False):
        """
        Returns the jump times at which the dimension does not change.

        Parameters:
            return_index: bool, optional
                If True, also return the indices of the non-dimension jump
                times.
        
        Returns:
            non_dimension_jump_times: list
                The times at which there are no dimension jumps.
        """
        jump_times, jump_indices = self.jump_times(return_index=True)
        dimension_jump_times, dimension_jump_indices = (
            self.dimension_jump_times(return_index=True))
        non_dimension_jump_times = jump_times[~np.isin(jump_times,
                                                       dimension_jump_times)]
        
        if return_index:
            non_dimension_jump_indices = jump_indices[~np.isin(
                jump_indices, dimension_jump_indices)]
            return non_dimension_jump_times, non_dimension_jump_indices
        else:
            return non_dimension_jump_times
        
    def plot_coordinates(self, title='Plot of the coordinates', legend=True,
                         labels=None, xlabel='Time', ylabel='Value',
                         separate_plots=False, sharex=True, sharey=True,
                         grid=False, show_jump_times=False,
                         show_dimension_jump_times=False):
        """
        Plots the coordinate values of the function against time.

        Parameters:
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            labels: list, optional
                Custom labels for the plot lines.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            separate_plots: bool, optional
                If True, create separate plots for each dimension.
            sharex: bool, optional
                If True, share the x-axis among separate plots.
            sharey: bool, optional
                If True, share the y-axis among separate plots.
            grid: bool, optional
                If True, show grid lines on the plot.
            show_jump_times: bool, optional
                If True, indicate jump times on the plot.
            show_dimension_jump_times: bool, optional
                If True, indicate dimension jump times on the plot.
        
        Returns:
            fig: The figure object created.
        """
        # Create figures
        fig = plt.figure()
        dimension_jump_times, dimension_jump_indices = (
            self.dimension_jump_times(return_index=True))
        dimensions = self.dimension()
        max_dimension = max(dimensions)
        prev_index = 0
        
        if not labels:
            labels = [f'Coordinate {i+1}' for i in range(max_dimension)]
            
        if separate_plots:
            figures = []
            for i in range(max_dimension):
                figures.append(plt.figure())
        
        # Plot the function iteratively between all dimension jump times
        for index in dimension_jump_indices:
            plt.figure(fig)
            plt.gca().set_prop_cycle(None)
            current_dimension = dimensions[prev_index]
            if current_dimension == max_dimension:
                lines = plt.plot(self.times[prev_index:index+1],
                                 self.values[prev_index:index+1])
            else:
                plt.plot(self.times[prev_index:index+1],
                         self.values[prev_index:index+1])
                
            if separate_plots:
                for i in range(current_dimension):
                    plt.figure(figures[i])
                    plt.gca().set_prop_cycle(None)
                    values = [value[i] for value 
                              in self.values[prev_index:index+1]]
                    plt.plot(self.times[prev_index:index+1], values)
                    
            prev_index = index + 1
        
        plt.figure(fig)
        plt.gca().set_prop_cycle(None)
        current_dimension = dimensions[prev_index]
        if current_dimension == max_dimension:
            lines = plt.plot(self.times[prev_index:], self.values[prev_index:])
        else:
            plt.plot(self.times[prev_index:], self.values[prev_index:])
        
        # Add specified design objects
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
            
        # Plot vertical lines for (dimension) jump times, if desired
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
                    plt.legend(lines + [vlines, dvlines],
                               labels + ['Non-dimension jump times',
                                         'Dimension jump times'])
                else:
                    plt.legend(lines + [dvlines],
                               labels + ['Dimension jump times'])
            
        # Add design objects to separate coordinate plots, if desired
        if separate_plots and (sharex or sharey or title or grid
                               or show_jump_times):
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
                    vlines = plt.vlines(jump_times, ylim[0], ylim[1], 'red',
                                        '--')
                    plt.ylim(ylim)
                    if legend:
                        plt.legend(lines + [vlines], labels + ['Jump times'])
                if show_dimension_jump_times:
                    jump_times = self.dimension_jump_times()
                    ylim = plt.ylim()
                    dvlines = plt.vlines(jump_times, ylim[0], ylim[1], 'green',
                                         '--')
                    plt.ylim(ylim)
                    if legend:
                        if show_jump_times:
                            plt.legend(lines + [vlines, dvlines],
                                       labels + ['Non-dimension jump times',
                                                 'Dimension jump times'])
                        else:
                            plt.legend(lines + [dvlines],
                                       labels + ['Dimension jump times'])
             
        # Plot dimension coordinates after last jump in separate plot
        if separate_plots:
            for i in range(current_dimension):
                plt.figure(figures[i])
                plt.gca().set_prop_cycle(None)
                values = [value[i] for value in self.values[prev_index:]]
                plt.plot(self.times[prev_index:], values)
            
        plt.show()
        
        return fig
    
    def plot_norm(self, order=2, title='Plot of the norm', legend=True,
                  labels=None, xlabel='Time', ylabel='Value', grid=False,
                  show_jump_times=False, show_dimension_jump_times=False):
        """
        Plots the norm of the function over time.

        Parameters:
            order: int or list, optional
                The order of the norm to plot. Default is the 2-norm.
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            labels: list, optional
                Custom labels for the plot lines.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            grid: bool, optional
                If True, show grid lines on the plot.
            show_jump_times: bool, optional
                If True, indicate jump times on the plot.
            show_dimension_jump_times: bool, optional
                If True, indicate dimension jump times on the plot.
        
        Returns:
            fig: The figure object created.
        """
        # Create figure
        fig = plt.figure()
        
        if not isinstance(order, (list, np.ndarray)):
            order_list = [order]
        else:
            order_list = order
        if not labels:
            labels = [f'{order} norm' for order in order_list]
        
        # Compute and plot norms
        lines = []
        for order in order_list:
            value_norms = self.norm(order=order)
            line, = plt.plot(self.times, value_norms)
            lines.append(line)
        
        # Add specified design objects
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
            
        # Add vertical lines for (dimension) jump times, if desired
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
                    plt.legend(lines + [vlines, dvlines],
                               labels + ['Non-dimension jump times',
                                         'Dimension jump times'])
                else:
                    plt.legend(lines + [dvlines],
                               labels + ['Dimension jump times'])
        
        return fig
    
    def norm(self, order=None, return_times=False):
        """
        Computes the norm of the function values.

        Parameters:
            order: int or None, optional
                The order of the norm to compute.
            return_times: bool, optional
                If True, return the corresponding times with the norms.
        
        Returns:
            value_norms: list
                The computed norms.
            times: list, optional
                The corresponding times, if requested.
        """
        value_norms = [float(la.norm(value, ord=order)) for value
                       in self.values]
        
        if return_times:
            return value_norms, self.times
        
        return value_norms
    
    def dimension(self, t=None):
        """
        Returns the dimension of the function at a given time or the
        dimensions at all times.

        Parameters:
            t: float, list, or None, optional
                The time point(s) at which to evaluate the dimension.
        
        Returns:
            dimension_values: list
                The dimensions at the requested times.
        
        Raises:
            TypeError: If t is not array-like nor a number.
        """
        # If t is not provided, return dimension for all times
        if t is None:
            dimension_values = [len(value) for value in self.values]
            return dimension_values
        
        # If t is a number, return the dimension at that time
        elif isinstance(t, (int, float, np.float16, np.float32, np.float64,
                            np.int16, np.int32, np.int64)):
            dimension_values = self.dimension()
            if t == self.times[-1]:
                return dimension_values[-1]
            index_t = bisect.bisect_right(self.times, t)
            dimension_t = dimension_values[index_t]
            return dimension_t
        
        # If t is array-like, return the dimensions at these times
        elif isinstance(t, (list, np.ndarray, tuple)):
            dimensions_t = [self.dimension(time) for time in t]
            return dimensions_t
        
        # If t is neither, raise an error
        else:
            raise TypeError(f'{t} is not array-like nor a number')
    
    def left_limit(self, t):
        """
        Returns the left limit of the function at a given time.

        Parameters:
            t: float
                The time at which to evaluate the left limit.
        
        Returns:
            value: The value of the left limit.
        """
        if t in self.jump_times():
            value = self.values[self.times.index(t)]
        else:
            value = self(t)
            
        return value
    
    def concatenate(self, other):
        """
        Concatenates this function with another function of variable
        dimension.

        Parameters:
            other: function_of_variable_dimension
                The function to concatenate with.
        
        Returns:
            func: function_of_variable_dimension
                The concatenated function.
        
        Raises:
            TypeError: If the other object is not a function of variable
                dimension.
            ValueError: If the dimensions of the functions do not match at
                overlap.
        """
        if not isinstance(other, FunctionOfVariableDimension):
            raise TypeError(f'{other} is not a function of variable dimension')
        
        # Handle the special case of an empty function
        if len(self.times) == 0:
            func = other
        elif len(other.times) == 0:
            func = self
            
        # If the last time of this function equals the first time of the
        # other function, append the times and values
        elif self.times[-1] == other.times[0]:
            func = FunctionOfVariableDimension(self.times + other.times,
                                                  self.values + other.values)
        
        # If the last time of this function is smaller than the first time of
        # the other function, append the times and values if the last and first
        # dimension match. Otherwise, raise an error.
        elif self.times[-1] < other.times[0]:
            if ((self.dimension(self.times[-1])
                != other.dimension(other.times[0]))
                and not other.times[0] in other.jump_times()):
                raise ValueError('Dimensions of function do not coincide (only\
                                 possible when last time for first function\
                                 and first time for second function are\
                                 the same)')
            if ((self.dimension(self.times[-1])
                != other.dimension(other.times[0]))
                and other.times[0] in other.jump_times()
                and self.dimension(self.times[-1])
                != len(other.left_limit(other.times[0]))):
                raise ValueError('Dimensions of function do not coincide (only\
                                 possible when last time for first function\
                                 and first time for second function are\
                                 the same)')
                
            func = FunctionOfVariableDimension(self.times + other.times,
                                                  self.values + other.values)
        
        # If both functions overlap, check if they coincide at overlapping
        # times. If not, raise an error, otherwise merge times and values.
        else:
            jump_position = 0
            for index_self, t in enumerate(self.times):
                if t >= other.times[0]:
                    if t in other.times:
                        index_other = other.times.index(t) + jump_position
                        if t in other.jump_times():
                            jump_position = 1 - jump_position
                        if (self.values[index_self]
                            != other.values[index_other]):
                            raise ValueError('Functions do not coincide at \
                                             overlapping times')
                    else:
                        if self.dimension(t) != other.dimension(t):
                            raise ValueError('Functions do not coincide at \
                                             overlapping times')
            
            func = FunctionOfVariableDimension(self.times + other.times,
                                                  self.values + other.values)
            
        return func
    
    
class StatsFunctionsOfVariableDimension():
    """
    A class to compute and store statistics of functions of variable
    dimension.

    Attributes:
        evaluated_functions: Count of the functions evaluated.
        times: A list of time points for evaluations.
        order: A list of orders for norms to be computed.
        norm_means: Means of the norms for each order over time.
        norm_variances: Variances of the norms for each order over time.
        norm_m2: Sum of squared differences of the norms for each order
            over time.
        dimension_means: Means of the dimensions over time.
        dimension_variances: Variances of the dimensions over time.
        dimension_m2: Sum of squared differences of the dimension for each
            order over time.
        custom_evaluation_functions: List of user-provided functions to
            evaluate.
        cef_means: Means of the custom evaluation functions over time.
        cef_variances: Variances of the custom evaluation functions over
            time.
        cef_m2: Sum of squared differences of the custom evaluation
            functions over time.
        collect_jump_times: Indicates whether jump times are stored.
        jump_times: List of all jump times of evaluated functions. Exists
            only, when collect_jump_times is True.
        collect_dimension_jump_times: Indicates whether dimension jump
            times are stored.
        dimension_jump_times: List of all dimension jump times of
            evaluated functions. Exists only, when
            collect_dimension_jump_times is True.
    """
    
    def __init__(self, t_span, stepsize=1, t_eval=None, order=2,
                 collect_jump_times=False, collect_dimension_jump_times=False):
        """
        Initializes the statistics collector.

        Parameters:
            t_span: tuple
                The start and end times for evaluations.
            stepsize: float, optional
                The step size for time evaluations.
            t_eval: list or array-like, optional
                Specific times at which to evaluate the functions.
            order: int, list or array-like, optional
                The order(s) of the norm(s) to compute.
            collect_jump_times: bool, optional
                If True, collect jump times during evaluation.
            collect_dimension_jump_times: bool, optional
                If True, collect dimension jump times during evaluation.
        """
        # Create self.times according to stepsize or t_eval
        t0, tf = t_span
        decimal_places = (len(str(stepsize).split('.')[1])
                          if '.' in str(stepsize) else 0)
        if t_eval is not None:
            self.times = self.__partial_t_eval__(t_eval, t0, tf)
        else:
            self.times = np.round(np.arange(t0, tf, stepsize), decimal_places)
        if not np.isclose(self.times[-1], tf):
            self.times = np.append(self.times, tf)
            
        # Initialize all other attributes
        self.evaluated_functions = 0
            
        if isinstance(order, (list, np.ndarray, tuple)):
            self.order = list(order)
        elif isinstance(order, (int, float, np.float16, np.float32, np.float64,
                                np.int16, np.int32, np.int64)):
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
        """
        Adds a function to the statistics collector for evaluation.

        Parameters:
            func: function_of_variable_dimension
                The function to add and evaluate.
        
        Raises:
            TypeError: If func is not a function of variable dimension.
            ValueError: If the function's time range does not match the
                collector's.
        """
        if not isinstance(func, FunctionOfVariableDimension):
            raise TypeError(f'{func} is not a function of variable dimension')
            
        if func.times[0] > self.times[0] or func.times[-1] < self.times[-1]:
            raise ValueError(f'{func} has values between {func.times[0]} and\
                             {func.times[-1]} but they need to be at least \
                                 between {self.times[0]} and {self.times[-1]}')
            
        self.evaluated_functions += 1
        
        values = func(self.times)

        # Update norm means and variances
        for index, order in enumerate(self.order):
            
            norm_values = np.array([np.linalg.norm(value, ord=order)
                                    for value in values])
            old_means = self.norm_means[index]
            old_m2 = self.norm_m2[index]
            
            new_means, new_variances, new_m2 = self._update_means_and_variances(
                norm_values, old_means, old_m2)
            
            self.norm_means[index] = new_means
            self.norm_m2[index] = new_m2
            self.norm_variances[index] = new_variances

        # Update dimension means and variances
        dimension_values = np.array(func.dimension(self.times))
        old_means = self.dimension_means
        old_m2 = self.dimension_m2
        
        new_means, new_variances, new_m2 = self._update_means_and_variances(
            dimension_values, old_means, old_m2)

        self.dimension_means = new_means
        self.dimension_m2 = new_m2
        self.dimension_variances = new_variances
        
        # Update means and variances of custom evaluation functions
        for index, evaluation_func in enumerate(
                self.custom_evaluation_functions):
            
            cef_values = np.array(evaluation_func(values))
            old_means = self.cef_means[index]
            old_m2 = self.cef_m2[index]
            
            new_means, new_variances, new_m2 = self._update_means_and_variances(
                cef_values, old_means, old_m2)
            
            self.cef_means[index] = new_means
            self.cef_m2[index] = new_m2
            self.cef_variances[index] = new_variances

        # Collect jump times, if desired
        if self.collect_jump_times:
            self.jump_times.append(func.jump_times())
        if self.collect_dimension_jump_times:
            self.dimension_jump_times.append(func.dimension_jump_times())
        
    def _update_means_and_variances(self, values, old_means, old_m2):
        """
        Updates the means and variances based on new values.
        
        Updates the means and variances based on new values using Welford's
        online algorithm.

        Parameters:
            values: array-like
                The new values for which to update statistics.
            old_means: array-like
                The current means before the update.
            old_m2: array-like
                The current M2 values before the update.
        
        Returns:
            tuple: Updated means, variances, and M2 values.
        """
        delta = values - old_means
        new_means = old_means + delta / self.evaluated_functions
        delta2 = values - new_means
        new_m2 = old_m2 + delta * delta2
        if self.evaluated_functions > 1:
            new_variances = new_m2 / (self.evaluated_functions - 1)
        else:
            new_variances = 0
        
        return new_means, new_variances, new_m2
        
    def plot_norm(self, order=2, sigma=2, labels=None,
                  title='Plot of mean norm', legend=True, xlabel='Time',
                  ylabel='Norm', grid=False):
        """
        Plots the mean norms of the collected functions.

        Parameters:
            order: int or list, optional
                The order(s) of the norm to plot.
            sigma: float, optional
                The factor for the variance shading.
            labels: list, optional
                Custom labels for the plot lines.
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            grid: bool, optional
                If True, show grid lines on the plot.
        
        Returns:
            fig: The figure object created.
            
        Raises:
            ValueError: 
        """
        fig = plt.figure()
        
        if not isinstance(order, (list, np.ndarray)):
            order_list = [order]
        else:
            order_list = order
        
        if not labels:
            labels = [f'{order} norm' for order in order_list]
        
        lines = []
        
        for order in order_list:
            if not order in self.order:
                raise ValueError('{order} is not in self.order and hence was \
                                 not computed')
            index = self.order.index(order)
            line, = plt.plot(self.times, self.norm_means[index])
            plt.fill_between(self.times, self.norm_means[index]
                             -sigma*self.norm_variances[index],
                             self.norm_means[index]
                             +sigma*self.norm_variances[index], alpha=.3)
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
        
    def plot_dimension(self, sigma=2, labels=None,
                       title='Plot of mean dimension', legend=False,
                       xlabel='Time', ylabel='Dimension', grid=False):
        """
        Plots the mean dimensions of the collected functions.

        Parameters:
            sigma: float, optional
                The factor for the variance shading.
            labels: list, optional
                Custom labels for the plot lines.
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            grid: bool, optional
                If True, show grid lines on the plot.
        
        Returns:
            fig: The figure object created.
        """
        fig = plt.figure()
        
        lines = []
        
        labels = ['Dimension']
        
        line, = plt.plot(self.times, self.dimension_means)
        plt.fill_between(self.times, self.dimension_means
                         -sigma*self.dimension_variances, self.dimension_means
                         +sigma*self.dimension_variances, alpha=.3)
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
    
    def plot_cefs(self, sigma=2, labels=None, title='Plot of cefs',
                  legend=False, xlabel='Time', ylabel='Value', grid=False):
        """
        Plots the means of the custom evaluation functions (CEFs).

        Parameters:
            sigma: float, optional
                The factor for the variance shading.
            labels: list, optional
                Custom labels for the plot lines.
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            grid: bool, optional
                If True, show grid lines on the plot.
        
        Returns:
            fig: The figure object created.
        """
        fig = plt.figure()
        
        lines = []
        
        labels = labels or ['Cef {i+1}' for i
                            in range(len(self.custom_evaluation_functions))]
        
        for index in range(len(self.custom_evaluation_functions)):
            line, = plt.plot(self.times, self.cef_means[index])
            plt.fill_between(self.times, self.cef_means[index]
                             -sigma*self.cef_variances[index], 
                             self.cef_means[index]
                             +sigma*self.cef_variances[index], alpha=.3)
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
        
    def plot_jump_times(self, number=None, density=False, separate=False,
                        separate_plots=False, title=None, legend=False,
                        xlabel=None, ylabel=None, grid=False):
        """
        Plots a histogram of the jump times.

        Parameters:
            number: list, optional
                Specific jump numbers to plot.
            density: bool, optional
                If True, normalize the histogram.
            separate: bool, optional
                If True, plot jumps separately.
            separate_plots: bool, optional
                If True, create separate plots for each jump time.
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            grid: bool, optional
                If True, show grid lines on the plot.
        
        Returns:
            fig: The figure object created.
            
        Raises:
            ValueError: If jump times have not been collected.
        """
        if not self.collect_jump_times:
            raise ValueError(f'Jump times have not been in collected in \
                             {self}')
            
        fig = plt.figure()
        
        if not number:
            max_jumps = max(len(jump_times) for jump_times in self.jump_times)
            number = np.arange(1, max_jumps+1)
        if isinstance(number, (list, np.ndarray, tuple)):
            if separate_plots or separate:
                for index in number:
                    index -= 1
                    jump_times = [item[index] for item in self.jump_times
                                  if index < len(item)]
                    plt.hist(jump_times, density=density,
                             label=f'Jump {index+1}')
                    if separate_plots:
                        plt.figure()
            else:
                jump_times = []
                for index in number:
                    index -= 1
                    jump_times.extend([item[index] for item in self.jump_times
                                       if index < len(item)])
                plt.hist(jump_times, density=density)
        elif isinstance(number, (int, float, np.float16, np.float32,
                                 np.float64, np.int16, np.int32, np.int64)):
            number -= 1
            jump_times = [item[int(number)] for item in self.jump_times
                          if int(number) < len(item)]
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
    
    def plot_inter_jump_times(self, number=None, density=False, separate=False,
                        separate_plots=False, title=None, legend=False,
                        xlabel=None, ylabel=None, grid=False):
        """
        Plots a histogram of the inter-jump times.

        Parameters:
            number: list, optional
                Specific jump numbers to plot.
            density: bool, optional
                If True, normalize the histogram.
            separate: bool, optional
                If True, create separate plots for each jump.
            separate_plots: bool, optional
                Create separate plots for each inter-jump time.
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            grid: bool, optional
                If True, show grid lines on the plot.
        
        Returns:
            fig: The figure object created.
                    
        Raises:
            ValueError: If jump times have not been collected.
        """
        if not self.collect_jump_times:
            raise ValueError(f'Jump times have not been in collected in \
                             {self}')
            
        fig = plt.figure()
        
        inter_jump_times = []

        for i, jump_times in enumerate(self.jump_times):
            inter_jump_times.append(np.zeros(len(jump_times)))
            if len(jump_times)>0:
                inter_jump_times[i][0] = jump_times[0]
                if len(jump_times)>1:
                    inter_jump_times[i][1:] = jump_times[1:] - jump_times[:-1]
            
        if not number:
            max_jumps = max(len(jump_times) for jump_times in self.jump_times)
            number = np.arange(1, max_jumps+1)
        if isinstance(number, (list, np.ndarray, tuple)):
            if separate_plots or separate:
                for index in number:
                    index -= 1
                    jump_times = [item[index] for item in inter_jump_times
                                  if index < len(item)]
                    plt.hist(jump_times, density=density,
                             label=f'Jump {index+1}')
                    if separate_plots:
                        plt.figure()
            else:
                jump_times = []
                for index in number:
                    index -= 1
                    jump_times.extend([item[index] for item in inter_jump_times
                                       if index < len(item)])
                plt.hist(jump_times, density=density)
        elif isinstance(number, (int, float, np.float16, np.float32,
                                 np.float64, np.int16, np.int32, np.int64)):
            number -= 1
            jump_times = [item[int(number)] for item in inter_jump_times
                          if int(number) < len(item)]
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
    
    def plot_number_of_jumps(self, density=False, title=None, legend=False,
                             xlabel=None, ylabel=None, grid=False):
        """
        Plots a histogram of the number of jumps.

        Parameters:
            density: bool, optional
                If True, normalize the histogram.
            title: str, optional
                The title of the plot.
            legend: bool, optional
                If True, show the legend.
            xlabel: str, optional
                Label for the x-axis.
            ylabel: str, optional
                Label for the y-axis.
            grid: bool, optional
                If True, show grid lines on the plot.
        
        Returns:
            fig: The figure object created.
                    
        Raises:
            ValueError: If jump times have not been collected.
        """
        if not self.collect_jump_times:
            raise ValueError(f'Jump times have not been in collected in \
                             {self}')

        number_of_jumps = [len(jump_times) for jump_times in self.jump_times]
        
        fig = plt.figure()
        
        plt.hist(number_of_jumps, density=density)
        
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
    
    def add_custom_evaluation_function(self, func):
        """
        Adds a custom evaluation function to be evaluated later.

        Parameters:
            func: callable
                The custom evaluation function to add.
        
        Raises:
            ValueError: If functions have already been evaluated.
        """
        if self.evaluated_functions > 0:
            raise ValueError('Can only add custom evaluation function if no \
                             functions have been evaluated yet')
            
        self.custom_evaluation_functions.append(func)
        
        self.cef_means.append(np.zeros(len(self.times)))
        self.cef_variances.append(np.zeros(len(self.times)))
        self.cef_m2.append(np.zeros(len(self.times)))
    
    def evaluate_norm(self, t):
        """
        Evaluates the mean norm at given time(s).

        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the mean norm.
        
        Returns:
            value: Evaluated norm means.
        """
        value = self._evaluate(t, self.norm_means)
        
        return value
    
    def evaluate_dimension(self, t):
        """
        Evaluates the mean dimension at a given time.

        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the mean dimension.
        
        Returns:
            value: Evaluated dimension means.
        """
        value = self._evaluate(t, self.dimension_means)
        
        return value
    
    def evaluate_cefs(self, t):
        """
        Evaluates the mean of the custom evaluation functions at given time(s).

        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the mean of the CEFs.
        
        Returns:
            value: Evaluated custom evaluation function means.
        """
        value = self._evaluate(t, self.cef_means)
        
        return value    
    
    def evaluate_norm_variance(self, t):
        """
        Evaluates the variances of the norms at given time(s).

        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the variances.
        
        Returns:
            value: Evaluated norm variances.
        """
        value = self._evaluate(t, self.norm_variances)
        
        return value
    
    def evaluate_dimension_variances(self, t):
        """
        Evaluates the variances of the dimensions at given time(s).

        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the variances.
        
        Returns:
            value: Evaluated dimension variances.
        """
        value = self._evaluate(t, self.dimension_variances)
        
        return value
        
    def evaluate_cef_variances(self, t):
        """
        Evaluates the variances of the cefs at given time(s).

        Evaluates the variances of the custom evaluation functions
        at given time(s).
        
        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the variances of the 
                CEFs.
        
        Returns:
            value: Evaluated variances of the CEFs.
        """
        value = self._evaluate(t, self.cef_variances)
        
        return value
    
    def _evaluate(self, t, values):
        """
        Evaluates the specified values at given time points by interpolation.

        Parameters:
            t: float, list, or array-like
                The time point(s) at which to evaluate the values.
            values: array-like
                The values to evaluate.
        
        Returns:
            value: Evaluated value(s) at the specified time point(s).
        """
        if isinstance(t, (list, np.ndarray)):
            value = []
            for time in t:
                value.append(self._interpolate_value(time, values))
        else:
            value = self._interpolate_value(t, values)
        
        return value

    def _interpolate_value(self, t, values):
        """
        Interpolates the values at a given time(s).

        Parameters:
            t: float
                The time at which to interpolate the value.
        
        Returns:
            value: float
                The interpolated value.
        
        Raises:
            ValueError: If the time is outside the range of defined times.
        """
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(f't is not between {self.times[0]} and \
                             {self.times[-1]} (first and last value of times)')
        elif t == self.times[-1]:
            return np.array(values[:,-1])
        
        index_t = bisect.bisect_right(self.times, t)
        
        interp_times = np.array(self.times[index_t-1:index_t+1])
        interp_values = np.array(values).T[index_t-1:index_t+1]
        
        value = interp_values[0] + (((t - interp_times[0]) 
                                    / (interp_times[1] - interp_times[0]))
                                    * (interp_values[1] - interp_values[0]))
        
        return value

    @staticmethod
    def __partial_t_eval__(t_eval, t0, tf):
        """
        Reduces t_eval to values between t0 and tf.
        
        Prepares the evaluation times by ensuring they are within the 
        specified range.

        Parameters:
            t_eval: list or array-like
                The specified evaluation times.
            t0: float
                The start of the time range.
            tf: float
                The end of the time range.

        Returns:
            t_eval_partial: array
                The filtered evaluation times within the range.
        """
        if t_eval is None:
            return None
        
        t_eval = np.array(t_eval)
        t_eval_partial = t_eval[(t_eval >= t0) & (t_eval <= tf)].tolist()
        
        if t0 not in t_eval_partial:
            t_eval_partial.insert(0, t0)
            
        if tf not in t_eval_partial:
            t_eval_partial.append(tf)
            
        t_eval_partial = np.array(t_eval_partial)

        return t_eval_partial


if __name__ == '__main__':
    times = [0, 1, 1, 2, 2, 4, 4, 5]
    values = [[0], [1], [.5, .5], [1.5, 2.5], [4], [6], [8], [9]]
    func = FunctionOfVariableDimension(times, values)
    jt, index = func.jump_times(return_index=True)
    djt = func.dimension_jump_times()
    ndjt = func.non_dimension_jump_times()
    print(jt, index)
    print(djt)
    print(ndjt)
    #func.delete_values(4)
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
    func = FunctionOfVariableDimension(times, values)
    func_2 = FunctionOfVariableDimension(2, 3)
    
    times = [0, 1, 1, 2, 2, 4]
    values = [[0], [2], [1, 1], [3, 5], [8], [12]]
    times_2 = [4, 5, 6]
    values_2 = [[2, 2, 2], [4, 3, 1], [6, 4, 0]]
    
    func_new = FunctionOfVariableDimension(times, values)
    func_2 = FunctionOfVariableDimension(times_2, values_2)
    
    func_conc = func.concatenate(func_2)
    
    func.plot_coordinates()
    func_new.plot_coordinates()
    func_2.plot_coordinates()
    func_conc.plot_coordinates()
    
    test = func([1.5, 3])
    print(test)
    
    stats = StatsFunctionsOfVariableDimension((0, 4), .25, order=[1, 2],
                                                  collect_jump_times=True)
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
    
    func = FunctionOfVariableDimension(times, values)
    fig = func.plot_coordinates(show_jump_times=True)
    plt.figure(fig)
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    plt.xlim(0, 2.5)
    plt.savefig('cadlag_function.png')
    plt.show()