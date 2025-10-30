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

import logging
from typing import Dict, List, Optional, Union

import sympy
import sympy.matrices
import sympy.utilities
import sympy.utilities.autowrap

from odetoolbox.config import Config
from odetoolbox.sympy_helpers import SymmetricEq, _sympy_parse_real

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
        self.all_variable_symbols = [sympy.Symbol(s, real=True) for s in self.all_variable_symbols]

        self.set_spike_times(spike_times)

        self.enable_caching = enable_caching
        self.enable_cache_update_ = True
        self.t = 0.

        #
        #   define the necessary numerical state variables
        #

        self.dim = len(self.all_variable_symbols)
        self.initial_values = {sympy.Symbol(k, real=True): v for k, v in self.solver_dict["initial_values"].items()}
        self.set_initial_values(self.initial_values)
        self.shape_starting_values = {sympy.Symbol(k, real=True): v for k, v in self.solver_dict["initial_values"].items()}
        for sym, v in self.shape_starting_values.items():
            expr = _sympy_parse_real(v, global_dict=Shape._sympy_globals)
            subs_dict = {}
            if "parameters" in self.solver_dict.keys():
                for parameter_name, v_ in self.solver_dict["parameters"].items():
                    parameter_symbol = sympy.Symbol(parameter_name, real=True)
                    subs_dict[parameter_symbol] = v_

            self.shape_starting_values[sym] = float(expr.evalf(subs=subs_dict))

        #
        #   initialise update expressions depending on whether conditional solver or not
        #

        if "update_expressions" in self.solver_dict.keys():
            self._pick_unconditional_solver()
        else:
            assert "conditions" in self.solver_dict.keys()
            self._pick_solver_based_on_condition()

        #
        #  reset the system to t = 0
        #

        self.reset()

    def _condition_holds(self, condition_string) -> bool:
        r"""Check boolean conditions of the form:

        ::

           (p_1 && p_2 .. && p_k) || (q_1 && q_2 .. && q_j) || ...

        """
        for sub_condition_string in condition_string.split("||"):
            # if any of the subterms hold, the whole expression holds (OR-ed together)
            if self._and_condition_holds(sub_condition_string):
                return True

        return False

    def _and_condition_holds(self, condition_string) -> bool:
        r"""Check boolean conditions of the form:

        ::

           p_1 && p_2 .. && p_k

        """
        sub_conditions = condition_string.split("&&")
        for sub_condition_string in sub_conditions:
            sub_condition_string = sub_condition_string.strip().strip("()")
            if "==" in sub_condition_string:
                parts = sub_condition_string.split("==")
            else:
                parts = sub_condition_string.split("!=")

            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()
            lhs = _sympy_parse_real(lhs_str, global_dict=Shape._sympy_globals)
            rhs = _sympy_parse_real(rhs_str, global_dict=Shape._sympy_globals)

            if "==" in sub_condition_string:
                equation = SymmetricEq(lhs, rhs)
            else:
                equation = sympy.Ne(lhs, rhs)

            subs_dict = {}
            if "parameters" in self.solver_dict.keys():
                for param_name, param_val in self.solver_dict["parameters"].items():
                    param_symbol = sympy.Symbol(param_name, real=True)
                    subs_dict[param_symbol] = param_val

            sub_condition_holds = equation.subs(subs_dict)

            if not sub_condition_holds:
                # if any of the subterms do not hold, the whole expression does not hold (AND-ed together)
                return False

        return True

    def _pick_unconditional_solver(self):
        self.update_expressions = self.solver_dict["update_expressions"].copy()
        self.propagators = self.solver_dict["propagators"].copy()
        self._process_update_expressions_from_solver_dict()

    def _pick_solver_based_on_condition(self):
        r"""In case of a conditional propagator solver: pick a solver depending on the conditions that hold (depending on parameter values)"""
        self.update_expressions = self.solver_dict["conditions"]["default"]["update_expressions"]
        self.propagators = self.solver_dict["conditions"]["default"]["propagators"]

        for condition, conditional_solver in self.solver_dict["conditions"].items():
            if condition != "default" and self._condition_holds(condition):
                self.update_expressions = conditional_solver["update_expressions"]
                self.propagators = conditional_solver["propagators"]
                logging.debug("Picking solver based on condition: " + str(condition))

                break

        self._process_update_expressions_from_solver_dict()

    def _process_update_expressions_from_solver_dict(self):
        #
        #   create substitution dictionary to replace symbolic variables with their numerical values
        #

        subs_dict = {}
        for prop_name, prop_expr in self.propagators.items():
            subs_dict[prop_name] = prop_expr

        if "parameters" in self.solver_dict.keys():
            for param_name, param_expr in self.solver_dict["parameters"].items():
                subs_dict[param_name] = param_expr

        # subs_dict = {sympy.Symbol(k, real=True): v for k, v in subs_dict.items()}
        subs_dict = {sympy.Symbol(k, real=True): v if type(v) is float or isinstance(v, sympy.Expr) else _sympy_parse_real(v, global_dict=Shape._sympy_globals) for k, v in subs_dict.items()}

        #
        #   parse the expressions from JSON if necessary
        #

        for k, v in self.update_expressions.items():
            if type(self.update_expressions[k]) is str:
                self.update_expressions[k] = _sympy_parse_real(self.update_expressions[k], global_dict=Shape._sympy_globals)

        #
        #   perform substitution in update expressions ahead of time to save time later
        #

        for k, v in self.update_expressions.items():
            for sym in self.update_expressions[k].free_symbols:
                assert sym.is_real
            self.update_expressions[k] = self.update_expressions[k].subs(subs_dict)
            for sym in self.update_expressions[k].free_symbols:
                assert sym.is_real
            self.update_expressions[k] = self.update_expressions[k].subs(subs_dict)
            for sym in self.update_expressions[k].free_symbols:
                assert sym.is_real

        #
        #    autowrap
        #

        self.update_expressions_wrapped = {}
        for k, v in self.update_expressions.items():
            self.update_expressions_wrapped[k] = sympy.utilities.autowrap.autowrap(v,
                                                                                   args=[sympy.Symbol(Config().output_timestep_symbol, real=True)] + self.all_variable_symbols,
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

    def set_initial_values(self, vals: Union[Dict[str, str], Dict[sympy.Symbol, sympy.Expr]]):
        r"""
        Set initial values, i.e. the state of the system at :math:`t = 0`. This will additionally cause the system state to be reset to :math:`t = 0` and the new initial conditions.

        :param vals: New initial values.
        """
        for sym, expr in vals.items():
            if type(sym) is str:
                sym = sympy.Symbol(sym, real=True)

            assert sym in self.initial_values.keys(), "Tried to set initial value for unknown parameter \"" + str(k) + "\""

            if type(expr) is str:
                expr = _sympy_parse_real(expr, global_dict=Shape._sympy_globals)

            subs_dict = {}
            if "parameters" in self.solver_dict.keys():
                for param_name, param_val in self.solver_dict["parameters"].items():
                    param_symbol = sympy.Symbol(param_name, real=True)
                    subs_dict[param_symbol] = param_val

            try:
                if type(expr) is float:
                    self.initial_values[sym] = expr
                else:
                    self.initial_values[sym] = float(expr.evalf(subs=subs_dict))
            except TypeError:
                msg = "Could not convert initial value expression to float. The following symbol(s) may be undeclared: " + ", ".join([str(expr_) for expr_ in expr.evalf(subs=subs_dict).free_symbols])
                raise Exception(msg)

        self.reset()

    def _update_step(self, delta_t, initial_values) -> Dict[sympy.Symbol, sympy.Expr]:
        r"""
        Apply propagator to update the state, starting from `initial_values`, by timestep `delta_t`.

        :param delta_t: Timestep to take.
        :param initial_values: A dictionary mapping variable names (as strings) to initial value expressions.
        :return new_state: A dictionary mapping symbols to state values.
        """

        new_state = {}

        #
        #    replace expressions by their numeric values
        #

        y = [delta_t] + [initial_values[sym] for sym in self.all_variable_symbols]

        #
        #    for each state variable, perform the state update
        #

        for state_variable, expr in self.update_expressions.items():
            new_state[sympy.Symbol(state_variable, real=True)] = self.update_expressions_wrapped[state_variable](*y)

        return new_state

    def get_value(self, t: float) -> Dict[sympy.Symbol, sympy.Expr]:
        r"""
        Get numerical solution of the dynamical system at time :python:`t`.

        :param t: The time to compute the solution for.
        :return state: A dictionary mapping symbols to state values.
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
