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

    def __init__(self, solver_dict, spike_times, enable_caching=True):
        """
        Parameters
        ----------
        spike_times : Dict[str, List[float]]
            for each variable symbol, list of times at which a spike occurs
        """

        self.solver_dict = solver_dict
        self.spike_times = spike_times
        self.enable_caching = enable_caching
        self.enable_cache_update_ = True
        self.t = 0.

        print("Initialised AnalyticIntegrator with spike times = " + str(spike_times))


        #
        #   define the necessary numerical state variables
        #

        self.dim = len(self.solver_dict["state_variables"])
        self.initial_values = self.solver_dict["initial_values"].copy()
        self.set_initial_values(self.initial_values)
        self.shape_starting_values = self.solver_dict["initial_values"].copy()
        for k, v in self.shape_starting_values.items():
            expr = sympy.parsing.sympy_parser.parse_expr(v, global_dict=Shape._sympy_globals)
            subs_dict = {}
            if "parameters" in self.solver_dict.keys():
                for k_, v_ in self.solver_dict["parameters"].items():
                    subs_dict[k_] = v_
            self.shape_starting_values[k] = float(expr.evalf(subs=subs_dict))

        self.update_expressions = self.solver_dict["update_expressions"].copy()
        for k, v in self.update_expressions.items():
            if type(self.update_expressions[k]) is str:
                self.update_expressions[k] = sympy.parsing.sympy_parser.parse_expr(self.update_expressions[k], global_dict=Shape._sympy_globals)


        #
        #  reset the system to t = 0
        #

        self.reset()


        #
        #   make a sorted list of all spike times for all symbols
        #

        self.all_spike_times = []
        self.all_spike_times_sym = []
        for sym, spike_times in self.spike_times.items():
            assert str(sym) in self.solver_dict["state_variables"], "Tried to set a spike time of unknown symbol \"" + str(sym) + "\""
            for t_sp in spike_times:
                if t_sp in self.all_spike_times:
                    idx = self.all_spike_times.index(t_sp)
                    self.all_spike_times_sym[idx].extend([sym])
                else:
                    self.all_spike_times.append(t_sp)
                    self.all_spike_times_sym.append([sym])

        idx = np.argsort(self.all_spike_times)
        self.all_spike_times = [ self.all_spike_times[i] for i in idx ]
        self.all_spike_times_sym = [ self.all_spike_times_sym[i] for i in idx ]
        print("Initialised AnalyticIntegrator with all_spike_times = " + str(self.all_spike_times))
        print("Initialised AnalyticIntegrator with all_spike_times_sym = " + str(self.all_spike_times_sym))


        #
        #   in the update expression, replace symbolic variables with their numerical values
        #

        self.subs_dict = {}
        for prop_symbol, prop_expr in self.solver_dict["propagators"].items():
            self.subs_dict[prop_symbol] = prop_expr
        if "parameters" in self.solver_dict.keys():
            for param_symbol, param_expr in self.solver_dict["parameters"].items():
                self.subs_dict[param_symbol] = param_expr

        #
        #   perform substtitution in update expressions ahead of time to save time later
        #

        print("subs_dict = " + str(self.subs_dict))
        for k, v in self.update_expressions.items():
            self.update_expressions[k] = self.update_expressions[k].subs(self.subs_dict).subs(self.subs_dict)


        #
        #    autowrap
        #

        self.update_expressions_wrapped = {}
        for k, v in self.update_expressions.items():
            print("Wrapping " + str(k) + " (expr = " + str(v) + ") with args " + str([sympy.Symbol("__h")] + [sympy.Symbol(sym) for sym in self.solver_dict["state_variables"]]))
            self.update_expressions_wrapped[k] = sympy.utilities.autowrap.autowrap(v, args=[sympy.Symbol("__h")] + [sympy.Symbol(sym) for sym in self.solver_dict["state_variables"]], backend="cython")


    def all_variable_symbols(self):
        return [sympy.Symbol(sym) for sym in self.solver_dict["state_variables"]]


    def enable_cache_update(self):
        self.enable_cache_update_ = True


    def disable_cache_update(self):
        self.enable_cache_update_ = False


    def reset(self):
        self.t_curr = 0.
        self.state_at_t_curr = self.initial_values.copy()
        print("**************** RESETTING (old t = " +str(self.t_curr) +") TO INITIAL_VALUES = " + str(self.initial_values))


    def get_variable_symbols(self):
        return self.initial_values.keys()


    def set_initial_values(self, vals):
        """Set initial values, i.e. the state of the system at t = 0. This will additionally cause the system state to be reset to t = 0.

        Parameters
        ----------
        vals : Dict(str -> float)
            New initial values.
        """
        for k, v in vals.items():
            k = str(k)
            assert k in self.initial_values.keys(), "Tried to set initial value for unknown parameter \"" + str(k) + "\""
            expr = sympy.parsing.sympy_parser.parse_expr(str(v), global_dict=Shape._sympy_globals)
            subs_dict = {}
            for param_symbol, param_val in self.solver_dict["parameters"].items():
                subs_dict[param_symbol] = param_val
            try:
                self.initial_values[k] = float(expr.evalf(subs=subs_dict))
            except TypeError:
                msg = "Could not convert initial value expression to float. The following symbol(s) may be undeclared: " + ", ".join([str(expr_) for expr_ in expr.evalf(subs=subs_dict).free_symbols])
                raise Exception(msg)
        self.reset()
        print("**************** SETTING INITIAL_VALUES = " + str(self.initial_values))


    def update_step(self, delta_t, initial_values, debug=True):
        #new_state = { k : np.nan for k in initial_values.keys() }
        new_state = {}

        #
        #    replace expressions by their numeric values
        #

        """self.subs_dict["__h"] = delta_t
        for state_variable2 in self.solver_dict["state_variables"]:
            self.subs_dict[state_variable2] = initial_values[state_variable2]"""
        y = [delta_t] + [initial_values[sym] for sym in self.solver_dict["state_variables"]]


        #
        #    for each state variable, perform the state update
        #

        for state_variable, expr in self.update_expressions.items():
            #expr = float(expr.evalf(subs=self.subs_dict))
            #new_state[state_variable] = expr
            new_state[state_variable] = self.update_expressions_wrapped[state_variable](*y)

        return new_state


    def get_value(self, t, debug=True):

        print("Analytic integrator: state requested at t = " + str(t) + ", last cached state = " + str(self.t_curr))

        if (not self.enable_caching) \
         or t < self.t_curr:
            #print("\treset state")
            self.reset()

        print("\tCurrent state: t_curr = " + str(self.t_curr) + ", state = " + str(self.state_at_t_curr))

        t_curr = self.t_curr
        state_at_t_curr = self.state_at_t_curr

        #
        #   process spikes between t_curr and t
        #

        for spike_t, spike_syms in zip(self.all_spike_times, self.all_spike_times_sym):

            if spike_t <= t_curr:
                continue

            if spike_t > t:
                break

            print("\tpropagating till next spike time: " + str(spike_t))


            #
            #   apply propagator to update the state from `t_curr` to `spike_t`
            #

            delta_t = spike_t - t_curr
            if delta_t > 0:
                state_at_t_curr = self.update_step(delta_t, state_at_t_curr)

            #
            #   delta impulse increment
            #

            for spike_sym in spike_syms:
                print("\t\tincrementing? " + str(spike_sym))
                spike_sym = str(spike_sym)
                if spike_sym.replace("'", "__d") in self.initial_values.keys():
                    state_at_t_curr[spike_sym.replace("'", "__d")] += self.shape_starting_values[spike_sym.replace("'", "__d")]
                    print("\t\tincrementing " + str(spike_sym.replace("'", "__d")) + " by " + str(self.shape_starting_values[spike_sym.replace("'", "__d")]))

            t_curr = spike_t

        #print("\tCurrent state: t_curr = " + str(self.t_curr) + ", state = " + str(self.state_at_t_curr))

        #
        #   update cache with the value at the last spike time (if we update with the value at the last requested time (`t`), we would accumulate roundoff errors)
        #

        if self.enable_cache_update_:
            self.t_curr = t_curr
            self.state_at_t_curr = state_at_t_curr

        #
        #   apply propagator to update the state from `t_curr` to `t`
        #

        print("\tpropagating till requested time: " + str(t))
        delta_t = t - t_curr
        if delta_t > 0:
            state_at_t_curr = self.update_step(delta_t, state_at_t_curr)
            t_curr = t

        return state_at_t_curr #self.state_at_t_curr
