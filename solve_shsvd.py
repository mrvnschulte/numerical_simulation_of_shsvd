# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:15:51 2025

@author: schulte
"""

from scipy.integrate import solve_ivp
import numpy as np
from function_vd import (function_of_variable_dimension,
                         stats_functions_of_variable_dimension)
from random import random
import matplotlib.pyplot as plt

def solve_hsvd(flow_function, transition_function, initial_value, t_span,
               jump_times=None, jump_set_function=None,
               jump_set_transition_function=None, t_eval=None, flow_args=None,
               transition_args=None, jump_set_args=None,
               jump_set_transition_args=None):
    
    if transition_args is not None:
        def transition_function(t, y, fun=transition_function):
            return fun(t, y, *transition_args)
        
    if flow_args is not None:
        def flow_function(t, y, fun=flow_function):
            return fun(t, y, *flow_args)
    
    if jump_set_function is not None:
        if jump_set_args is not None:
            def jump_set_function(t, y, fun=jump_set_function):
                return fun(t, y, *jump_set_args)
        jump_set_function.terminal=True
        if not hasattr(jump_set_function, 'direction'):
            jump_set_function.direction = 0
        if jump_set_transition_function is None:
            jump_set_transition_function = transition_function
        elif jump_set_transition_args is not None:
            def jump_set_transition_function(t, y, 
                                             fun=jump_set_transition_function):
                return fun(t, y, *jump_set_transition_args)
        
    t0, end_time = t_span
    
    if isinstance(jump_times, (list, np.ndarray)):
        jump_times = list(jump_times)
        jump_times.sort()
        if jump_times[-1] < t0:
            jump_times = [end_time]
        else:
            while jump_times[0] < t0:
                del jump_times[0]
        tf = min(end_time, jump_times[0])
    else:
        tf = end_time
        
    func = function_of_variable_dimension()
    
    while tf < end_time:
    
        t_eval_partial = __partial_t_eval__(t_eval, t0, tf)
        y = solve_ivp(flow_function, (t0, tf), initial_value,
                      t_eval=t_eval_partial, events=jump_set_function)
        func.add_values(y.t, y.y.T)
        
        if jump_set_function is not None:
            if y.t_events[0].size > 0:
                t0 = y.t_events[0][0]
                y_jump = y.y_events[0][0]
                initial_value = jump_set_transition_function(t0, y_jump)
                if t0 not in y.t:
                    func.add_values(t0, y_jump)
            else:
                initial_value = transition_function(y.t[-1], y.y.T[-1])
                t0 = tf
                del jump_times[0]
                if len(jump_times) == 0:
                    tf = end_time
                else:
                    tf = min(end_time, jump_times[0])
 
        else:
            initial_value = transition_function(y.t[-1], y.y.T[-1])
            t0 = tf
            del jump_times[0]
            if len(jump_times) == 0:
                tf = end_time
            else:
                tf = min(end_time, jump_times[0])
        
    t_eval_partial = __partial_t_eval__(t_eval, t0, tf)
    y = solve_ivp(flow_function, (t0, tf), initial_value,
                  t_eval=t_eval_partial, events=jump_set_function)
    func.add_values(y.t, y.y.T)
    
    if jump_set_function is not None:
        
        while y.t_events[0].size > 0:
            
            initial_value = jump_set_transition_function(y.t[-1], y.y.T[-1])
            t0 = y.t_events[0][0]
            y_jump = y.y_events[0][0]
            if t0 not in y.t:
                func.add_values(t0, y_jump)


            t_eval_partial = __partial_t_eval__(t_eval, t0, tf)
            y = solve_ivp(flow_function, (t0, tf), initial_value,
                          t_eval=t_eval_partial, events=jump_set_function)
            func.add_values(y.t, y.y.T)
            
            
    return func


def solve_shsvd_single_run(flow_function, jump_rate_function,
                           transition_function, initial_value, t_span,
                           jump_rate_bound, jump_times=None,
                           jump_time_transition_function=None,
                           jump_set_function=None,
                           jump_set_transition_function=None, t_eval=None,
                           flow_args=None, jump_rate_args=None,
                           transition_args=None,
                           jump_time_transition_args=None, 
                           jump_set_args=None, jump_set_transition_args=None):
    
    if jump_rate_args is not None:
        def jump_rate_function(t, y, fun=jump_rate_function):
            return fun(t, y, *jump_rate_args)
        
    if transition_args is not None:
        def transition_function(t, y, fun=transition_function):
            return fun(t, y, *transition_args)
        
    if jump_times is not None and jump_time_transition_function is None:
        jump_time_transition_function = transition_function
            
    t0, end_time = t_span
        
    func = function_of_variable_dimension()
    
    r = random()
    t = t0 - np.log(r) / jump_rate_bound
    
    store_t0 = True
    
    while t < end_time:
    
        func_partial = solve_hsvd(flow_function, jump_time_transition_function,
                                  initial_value, (t0, t),
                                  jump_times=jump_times, t_eval=t_eval, 
                                  flow_args=flow_args, 
                                  transition_args=jump_time_transition_args,
                                  jump_set_args=jump_set_args,
                                  jump_set_transition_args=
                                  jump_set_transition_args)
        
        if not store_t0:
            func_partial.delete_values(times=[t0])
            
        last_time = func_partial.times[-1]
        last_value = func_partial.values[-1]
                
        t0 = t
        
        s = random()
        if s <= jump_rate_function(last_time, last_value)/jump_rate_bound:
            initial_value = transition_function(last_time, last_value)
            store_t0 = True
        else:
            initial_value = last_value
            if t_eval is not None:
                if last_time not in t_eval:
                    func_partial.delete_values()
                    store_t0 = False
        
        func = func.concatenate(func_partial)

        r = random()
        t = t - np.log(r) / jump_rate_bound
        
    func_partial = solve_hsvd(flow_function, jump_time_transition_function,
                              initial_value, (t0, end_time),
                              jump_times=jump_times, t_eval=t_eval,
                              flow_args=flow_args,
                              transition_args=jump_time_transition_args,
                              jump_set_args=jump_set_args,
                              jump_set_transition_args=
                              jump_set_transition_args)
    
    if not store_t0:
        func_partial.delete_values(times=[t0])

    func = func.concatenate(func_partial)
    
    return func


def solve_shsvd(flow_function, jump_rate_function, transition_function,
                initial_value, t_span, jump_rate_bound, iterations=500,
                jump_times=None, jump_time_transition_function=None,
                jump_set_function=None, jump_set_transition_function=None, 
                custom_evaluation_functions=None, stepsize=1, t_eval=None,
                norm_order=2, collect_jump_times=False,
                collect_dimension_jump_times=False, flow_args=None,
                jump_rate_args=None, transition_args=None,
                jump_time_transition_args=None, jump_set_args=None,
                jump_set_transition_args=None, 
                cef_args=None):
    
    stats = stats_functions_of_variable_dimension(t_span, stepsize,
                                                  t_eval=t_eval,
                                                  order=norm_order,
                                                  collect_jump_times=
                                                  collect_jump_times,
                                                  collect_dimension_jump_times=
                                                  collect_dimension_jump_times)
    
    if custom_evaluation_functions is not None:
        
        if isinstance(custom_evaluation_functions, list):
            
            if cef_args is not None:
                
                if len(custom_evaluation_functions) != len(cef_args):
                    raise ValueError('''Number of cef functions does not match
                                     number of cef arguments''')
                
                for cef, cef_arg in zip(custom_evaluation_functions, cef_args):

                    if cef_arg is not None:
                        def cef_with_args(values, fun=cef, args=cef_arg):
                            return fun(values, *args)
                        stats.add_custom_evaluation_function(cef_with_args)
                        
                    else:
                        stats.add_custom_evaluation_function(cef)
                
            else:
                for cef in custom_evaluation_functions:
                    stats.add_custom_evaluation_function(cef)
                
        else:
            stats.add_custom_evaluation_function(custom_evaluation_functions)
    
    for it in range(iterations):
        func = solve_shsvd_single_run(flow_function, jump_rate_function,
                                      transition_function, initial_value,
                                      t_span, jump_rate_bound,
                                      jump_times=jump_times,
                                      jump_time_transition_function=
                                      jump_time_transition_function,
                                      jump_set_function=jump_set_function,
                                      jump_set_transition_function=
                                      jump_set_transition_function,
                                      t_eval=t_eval, flow_args=flow_args, 
                                      jump_rate_args=jump_rate_args,
                                      transition_args=transition_args,
                                      jump_time_transition_args=
                                      jump_time_transition_args,
                                      jump_set_args=jump_set_args,
                                      jump_set_transition_args=
                                      jump_set_transition_args)
        stats.add_function(func)
    
    return stats

def __partial_t_eval__(t_eval, t0, tf):
    
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
    
    def flow_function(t, y):
        
        return -y
    
    def harmonic_oscillator(t, y, a, b):
        
        if len(y) == 1:
            dy = [0]
        elif len(y) == 2:
            dy = [y[1], -b * y[1] - a * y[0]]
        else:
            raise ValueError('Not a possible dimension')
            
        return dy

    def jump_rate_function(t, y):
        
        norm = np.linalg.norm(np.array(y))
        
        prob = min(1, 1/norm)
        
        return prob
    
    def ho_jump_rate_function(t, y):
        
        return np.abs(y[0])#np.linalg.norm(y[0])

    def transition_function(t, y):
        
        if len(y) == 2:
            y_new = [y[0] + y[1]]
        elif len(y) == 1:
            y_new = [1/3*y[0], 2/3*y[0]]
        else:
            raise ValueError('Not the right dimension')
        
        return y_new
    
    def ho_transition_function(t, y, a):
        
        if len(y) == 1:
            y_new = [y[0], a]
        elif len(y) == 2:
            y_new = [y[0]]
        else:
            raise ValueError('Not a possible dimension')
            
        return y_new
    
    def ho_transition_function_2(t, y, c):
        
        if len(y) == 1:
            y_new = [y[0], c*y[0]]
        elif len(y) == 2:
            y_new = [y[0]]
        else:
            raise ValueError('Not a possible dimension')
            
        return y_new
    
    
    def ho_energy(values, mass):
        
        squared_values = [np.array(value)**2 for value in values]
        energy = [1/2*mass*np.sum(value) for value in squared_values]
        
        return energy
    
    
    def ho_first_coordinate(values):
        
        first_coordinate = [value[0] for value in values]
        
        return first_coordinate
    
    def jump_set_function(t, y, b):
        
        return y[0] + b
    
    def jump_set_transition_function(t, y, c):
        
        return [c]
        
    
    # y0 = np.array([1, 2])
    # t_span = (0, 10)

    # func_det = solve_hsvd(flow_function, transition_function, y0, t_span)#, jump_times=[-1, 1, 5, 7])
    # func_det.plot_coordinates('Coordinate plot', True, ['first', 'second'], 'Time', 'Value', True, grid=True, show_jump_times=True, show_dimension_jump_times=True)
    # func_det.plot_norm(order=[1, 2], title='Norm plot', legend=True, grid=True, show_dimension_jump_times=True)
    
    # func_stoch = solve_shsvd_single_run(flow_function, jump_rate_function, transition_function, y0, t_span, 1)#, jump_times=[-1, 1, 5, 7])
    # func_stoch.plot_coordinates('Coordinate plot', True, ['first', 'second'], 'Time', 'Value', True, grid=True, show_jump_times=True, show_dimension_jump_times=True)
    # func_stoch.plot_norm(order=[1, 2], title='Norm plot', legend=True, grid=True, show_dimension_jump_times=True)
    
    # stats = solve_shsvd(flow_function, jump_rate_function, transition_function, y0, t_span, 1, norm_order=[1, 2], collect_jump_times=True)
    
    t_end = 100
    y0 = [1, 0]
    t_span = (0, t_end)
    jump_rate_bound = np.linalg.norm(y0, ord=2)
    a = 1
    b = 0
    transition_args = (-.3, )
    jump_times = np.arange(8.5, 100, 8.5)
    t_eval = np.linspace(0, t_end, 2*t_end+1)
    jump_set_args = (.1, )
    jump_set_transition_args = (.5, )
    
    jump_times = [20, 60]
    
    det_oscillator = solve_hsvd(harmonic_oscillator, ho_transition_function_2,
                                y0, t_span,
                                jump_set_function=jump_set_function,
                                jump_set_transition_function=
                                jump_set_transition_function,
                                flow_args=(a, b), 
                                transition_args=transition_args,
                                jump_times=jump_times, t_eval=t_eval,
                                jump_set_args=jump_set_args,
                                jump_set_transition_args=
                                jump_set_transition_args)    
    fig = det_oscillator.plot_coordinates()
    
    # plt.figure(fig)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('Harmonic_oscillator_hold_and_push.png')
    
    oscillator = solve_shsvd_single_run(harmonic_oscillator,
                                        ho_jump_rate_function,
                                        ho_transition_function_2, y0, t_span,
                                        jump_rate_bound, jump_times=jump_times,
                                        flow_args=(a, b),
                                        transition_args=transition_args,
                                        t_eval=t_eval)
    oscillator.plot_coordinates()
    oscillator_stats = solve_shsvd(harmonic_oscillator, ho_jump_rate_function,
                                   ho_transition_function_2, y0, t_span,
                                   jump_rate_bound, t_eval=t_eval,
                                   jump_times=jump_times, norm_order=[1, 2],
                                   collect_jump_times=True, flow_args=(a, b),
                                   transition_args=transition_args,
                                   iterations=50,
                                   custom_evaluation_functions=[ho_energy,
                                                                ho_first_coordinate],
                                   cef_args=[(2, ), None])
    oscillator_stats.plot_norm()