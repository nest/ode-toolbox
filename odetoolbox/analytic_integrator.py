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

from odetoolbox.sympy_helpers import _is_zero

from .shapes import Shape
from .integrator import Integrator
from .base_analytic_integrator import BaseAnalyticIntegrator


class AnalyticIntegrator(Integrator):
    r"""
    Integrate a dynamical system by means of the propagators returned by ODE-toolbox.

    This integrator also supports the use of conditions in the returned analytic integrators dictionary.
    """

    def __init__(self, solver_dicts, spike_times: Optional[Dict[str, List[float]]] = None, enable_caching: bool = True):
        r"""
        :param solve_dict: The results dictionary returned by a call to :python:`odetoolbox.analysis()`.
        :param spike_times: For each variable, used as a key, the list of times at which a spike occurs.
        :param enable_caching: Allow caching of results between requested times.
        """

        self.analytic_integrators = []
        for solver_dict in solver_dicts:
            self.analytic_integrators.append(BaseAnalyticIntegrator(solver_dict, spike_times, enable_caching=enable_caching))

        super(AnalyticIntegrator, self).__init__(spike_times)


    def get_all_variable_symbols(self):
        # variable symbols are the same for each AnalyticIntegrator instance; arbitrarily return the ones from integrator 0
        return self.analytic_integrators[0].all_variable_symbols


    def enable_cache_update(self):
        r"""
        Allow caching of results between requested times.
        """
        for analytic_integrator in self.analytic_integrators:
            analytic_integrator.enable_cache_update()


    def disable_cache_update(self):
        r"""
        Disallow caching of results between requested times.
        """
        for analytic_integrator in self.analytic_integrators:
            analytic_integrator.disable_cache_update()


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
        for analytic_integrator in self.analytic_integrators:
            analytic_integrator.set_initial_values(vals)


    def get_value(self, t):
        r"""
        Get numerical solution of the dynamical system at time :python:`t`.

        :param t: The time to compute the solution for.
        """

        #
        #   find out which analytic solver to use
        #

        if len(self.analytic_integrators) == 1:
            # there is only one
            analytic_integrator = self.analytic_integrator[0]
        else:
            analytic_integrator = None

            # pick the solver for which the condition holds
            for analytic_integrator_ in self.analytic_integrators:
                assert "condition" in analytic_integrator_.keys()
                if not analytic_integrator_["condition"] == "otherwise":
                    if _is_zero(sympy.eval(analytic_integrator["condition"])):
                        analytic_integrator = analytic_integrator_
                        break

            # none of the conditions holds, use the default/"otherwise" solver
            if analytic_integrator is None:
                for analytic_integrator_ in self.analytic_integrators:
                    assert "condition" in analytic_integrator_.keys()
                    if analytic_integrator_["condition"] == "otherwise":
                        analytic_integrator = analytic_integrator_
                        break

            assert not analytic_integrator is None

        return analytic_integrator.get_value(t)
