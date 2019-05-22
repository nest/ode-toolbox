#
# analytic_integrator.py
#
# This file is part of the NEST ODE toolbox.
#
# Copyright (C) 2017 The NEST Initiative
#
# The NEST ODE toolbox is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 2 of
# the License, or (at your option) any later version.
#
# The NEST ODE toolbox is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.
#

import sympy
import sympy.matrices
import numpy as np

from .shapes import Shape


class AnalyticIntegrator():
    """integrate a dynamical system by means of the propagators returned by odetoolbox"""
    
    def __init__(self, solver_dict, spike_times):
        """
        Parameters
        ----------
        spike_times : Dict[str, List[float]]
            for each variable symbol, list of times at which a spike occurs
        """
        
        self.solver_dict = solver_dict
        self.spike_times = spike_times
        
        self.t = 0.
        

        #
        #   define the necessary numerical state variables
        #

        self.dim = len(self.solver_dict["state_variables"])
        self.initial_values = self.solver_dict["initial_values"].copy()
        self.shape_starting_values = self.solver_dict["initial_values"].copy()
        for k, v in self.shape_starting_values.items():
            expr = sympy.parsing.sympy_parser.parse_expr(v)
            if "parameters" in self.solver_dict.keys():
                for k_, v_ in self.solver_dict["parameters"].items():
                    expr = expr.subs(k_, v_)
            self.shape_starting_values[k] = float(expr.evalf())

        self.update_expressions = self.solver_dict["update_expressions"].copy()
        for k, v in self.update_expressions.items():
            if type(self.update_expressions[k]) is str:
                self.update_expressions[k] = sympy.parsing.sympy_parser.parse_expr(self.update_expressions[k])

        #
        #   make a sorted list of all spike times for all symbols
        #

        self.all_spike_times = []
        self.all_spike_times_sym = []
        for sym in self.spike_times.keys():
            for t_sp in self.spike_times[sym]:
                if t_sp in self.all_spike_times:
                    idx = self.all_spike_times.index(t_sp)
                    self.all_spike_times_sym[idx].extend([sym])
                else:
                    self.all_spike_times.append(t_sp)
                    self.all_spike_times_sym.append([sym])

        idx = np.argsort(self.all_spike_times)
        self.all_spike_times = [ self.all_spike_times[i] for i in idx ]
        self.all_spike_times_sym = [ self.all_spike_times_sym[i] for i in idx ]
        
        #self.state_vec = np.nan * np.ones(self.dim)
        #self.prop_matrix = 



    def get_variable_symbols(self):
        return self.initial_values.keys()

    
    
    def set_initial_values(self, vals):
        """
        Parameters
        ----------
        vals : Dict(str -> float)
            New values
        """
        for k, v in vals.items():
            assert k in self.initial_values.keys(), "Tried to set initial value for unknown parameter \"" + str(k) + "\""
            self.initial_values[k] = v


    def update_step(self, delta_t, initial_values, debug=True):
        
        new_state = { k : np.nan for k in initial_values.keys() }
        
        for state_variable, expr in self.update_expressions.items():

            #if debug:
                ##print("\t* state_variable_name = " + state_variable)
                #print("\t* update expression = " + str(expr))

            #
            #   in the update expression, replace symbolic variables with their numerical values
            #

            for prop_symbol, prop_val in self.solver_dict["propagators"].items():
                expr = expr.subs(prop_symbol, prop_val)
            if "parameters" in self.solver_dict.keys():
                for param_symbol, param_val in self.solver_dict["parameters"].items():
                    expr = expr.subs(param_symbol, param_val)
            for state_variable2 in self.solver_dict["state_variables"]:
                expr = expr.subs(state_variable2, initial_values[state_variable2]) 
            expr = expr.subs("__h", delta_t)
            expr = float(expr.evalf())
            
            #if debug:
                #print("\t* update expression evaluates to = " + str(expr))

            new_state[state_variable] = expr

        return new_state


    def get_value(self, t, debug=True):
        #idx = np.where(self.all_spike_times > t)[0][0]
        #all_spike_times = self.all_spike_times[:idx]
        #all_spike_times_syms = self.all_spike_times_sym[:idx]
        
        #all_spike_times = []
        #all_spike_times_sym = []
        #for sym, sym_spikes in self.spike_times.items():
            #all_spike_times.extend(sym_spikes)
            #all_spike_times_sym.extend(len(sym_spikes) * [sym])

        #idx = np.argsort(all_spike_times)[0]
        #all_spike_times = [ all_spike_times[i] for i in idx ] 
        #all_spike_times_sym = [ all_spike_times_sym[i] for i in idx ]
        
        state_at_t_curr = self.initial_values.copy()
        t_curr = 0.
        
        for spike_t, spike_syms in zip(self.all_spike_times, self.all_spike_times_sym):
            
            if spike_t > t:
                break
            
            #
            #   apply propagator to update the state from `t_curr` to `spike_t`
            #
            
            delta_t = spike_t - t_curr
            assert delta_t > 0
            state_at_t_curr = self.update_step(delta_t, state_at_t_curr)

            #
            #   delta impulse increment
            #
            
            for spike_sym in spike_syms:
                state_at_t_curr[spike_sym] += self.shape_starting_values[spike_sym]
            
            t_curr = spike_t

            
        #
        #   apply propagator to update the state from `t_curr` to `t`
        #
        
        delta_t = t - t_curr
        state_at_t_curr = self.update_step(delta_t, state_at_t_curr)
        t_curr = t

        return state_at_t_curr
