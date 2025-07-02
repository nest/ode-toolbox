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

from typing import Dict, List, Optional

import sympy
import sympy.matrices
import sympy.utilities
import sympy.utilities.autowrap

from .shapes import Shape
from .integrator import Integrator


class AnalyticIntegrator(Integrator):
    r"""
    Integrate a dynamical system by means of the propagators returned by ODE-toolbox.
    """

    def __init__(self, solver_dict, spike_times: Optional[Dict[str, List[float]]] = None, enable_caching: bool = True):
        r"""
        :param solve_dict: The results dictionary returned by a call to :python:`odetoolbox.analysis()`.
        :param spike_times: For each variable, used as a key, the list of times at which a spike occurs.
        :param enable_caching: Allow caching of results between requested times.
        """

        super(AnalyticIntegrator, self).__init__()

        self.solver_dict = solver_dict

        self.all_variable_symbols = self.solver_dict["state_variables"]
        self.all_variable_symbols = [sympy.Symbol(s) for s in self.all_variable_symbols]

        self.set_spike_times(spike_times)

        self.enable_caching = enable_caching
        self.enable_cache_update_ = True
        self.t = 0.


        #
        #   define the necessary numerical state variables
        #

        self.dim = len(self.all_variable_symbols)
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
        #   in the update expression, replace symbolic variables with their numerical values
        #

        self.subs_dict = {}
        for prop_symbol, prop_expr in self.solver_dict["propagators"].items():
            self.subs_dict[prop_symbol] = prop_expr
        if "parameters" in self.solver_dict.keys():
            for param_symbol, param_expr in self.solver_dict["parameters"].items():
                self.subs_dict[param_symbol] = param_expr


        #
        #   perform substitution in update expressions ahead of time to save time later
        #

        for k, v in self.update_expressions.items():
            self.update_expressions[k] = self.update_expressions[k].subs(self.subs_dict).subs(self.subs_dict)

        #
        #    autowrap
        #

        self.update_expressions_wrapped = {}
        for k, v in self.update_expressions.items():
            self.update_expressions_wrapped[k] = sympy.utilities.autowrap.autowrap(v,
                                                                                   args=[sympy.Symbol("__h")] + self.all_variable_symbols,
                                                                                   backend="cython",
                                                                                   helpers=Shape._sympy_autowrap_helpers)


    def get_all_variable_symbols(self):
        return self.all_variable_symbols


    def enable_cache_update(self):
        r"""
        Allow caching of results between requested times.
        """
        self.enable_cache_update_ = True


    def disable_cache_update(self):
        r"""
        Disallow caching of results between requested times.
        """
        self.enable_cache_update_ = False


    def reset(self):
        r"""
        Reset time to zero and state to initial values.
        """
        self.t_curr = 0.
        self.state_at_t_curr = self.initial_values.copy()


    def set_initial_values(self, vals):
        r"""
        Set initial values, i.e. the state of the system at :math:`t = 0`. This will additionally cause the system state to be reset to :math:`t = 0` and the new initial conditions.

        :param vals: New initial values.
        """
        for k, v in vals.items():
            k = str(k)
            assert k in self.initial_values.keys(), "Tried to set initial value for unknown parameter \"" + str(k) + "\""
            expr = sympy.parsing.sympy_parser.parse_expr(str(v), global_dict=Shape._sympy_globals)
            subs_dict = {}
            if "parameters" in self.solver_dict.keys():
                for param_symbol, param_val in self.solver_dict["parameters"].items():
                    subs_dict[param_symbol] = param_val
            try:
                self.initial_values[k] = float(expr.evalf(subs=subs_dict))
            except TypeError:
                msg = "Could not convert initial value expression to float. The following symbol(s) may be undeclared: " + ", ".join([str(expr_) for expr_ in expr.evalf(subs=subs_dict).free_symbols])
                raise Exception(msg)
        self.reset()


    def _update_step(self, delta_t, initial_values):
        r"""
        Apply propagator to update the state, starting from `initial_values`, by timestep `delta_t`.

        :param delta_t: Timestep to take.
        :param initial_values: A dictionary mapping variable names (as strings) to initial value expressions.
        """

        new_state = {}

        #
        #    replace expressions by their numeric values
        #

        y = [delta_t] + [initial_values[str(sym)] for sym in self.all_variable_symbols]


        #
        #    for each state variable, perform the state update
        #

        for state_variable, expr in self.update_expressions.items():
            new_state[state_variable] = self.update_expressions_wrapped[state_variable](*y)

        return new_state


    def get_value(self, t):
        r"""
        Get numerical solution of the dynamical system at time :python:`t`.

        :param t: The time to compute the solution for.
        """

        if (not self.enable_caching) \
           or t < self.t_curr:
            self.reset()

        t_curr = self.t_curr
        state_at_t_curr = self.state_at_t_curr

        #
        #   grab stimulus spike times
        #

        all_spike_times, all_spike_times_sym = self.get_sorted_spike_times()


        #
        #   process spikes between ⟨t_curr, t]
        #

        for spike_t, spike_syms in zip(all_spike_times, all_spike_times_sym):

            if spike_t <= t_curr:
                continue

            if spike_t > t:
                break


            #
            #   apply propagator to update the state from `t_curr` to `spike_t`
            #

            delta_t = spike_t - t_curr
            if delta_t > 0:
                state_at_t_curr = self._update_step(delta_t, state_at_t_curr)
                t_curr = spike_t


            #
            #   delta impulse increment
            #

            for spike_sym in spike_syms:
                if spike_sym in self.initial_values.keys():
                    state_at_t_curr[spike_sym] += self.shape_starting_values[spike_sym]


        #
        #   update cache with the value at the last spike time (if we update with the value at the last requested time (`t`), we would accumulate roundoff errors)
        #

        if self.enable_cache_update_:
            self.t_curr = t_curr
            self.state_at_t_curr = state_at_t_curr


        #
        #   apply propagator to update the state from `t_curr` to `t`
        #

        delta_t = t - t_curr
        if delta_t > 0:
            state_at_t_curr = self._update_step(delta_t, state_at_t_curr)
            t_curr = t

        return state_at_t_curr
