#
# stiffness.py
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
import numpy as np
import numpy.random
import sympy

from .mixed_integrator import MixedIntegrator
from .mixed_integrator import ParametersIncompleteException
from .shapes import Shape
from .spike_generator import SpikeGenerator


try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError as ie:
    logging.warning("PyGSL is not available. The stiffness test will be skipped.")
    logging.warning("Error when importing: " + str(ie))
    PYGSL_AVAILABLE = False


class StiffnessTester:

    def __init__(self, system_of_shapes, shapes, analytic_solver_dict=None, parameters=None, stimuli=None, random_seed=123, max_step_size=np.inf, integration_accuracy_abs=1E-6, integration_accuracy_rel=1E-6, sim_time=100., alias_spikes=False):
        r"""
        :param system_of_shapes: Dynamical system to solve.
        :param shapes: List of shapes in the dynamical system.
        :param analytic_solver_dict: Analytic solver dictionary from ODE-toolbox analysis result.
        :param parameters: Dictionary mapping parameter name (as string) to value expression.
        :param stimuli: Dictionary containing spiking stimuli.
        :param random_seed: Random number generator seed.
        :param max_step_size: The maximum step size taken by the integrator.
        :param integration_accuracy_abs: Absolute integration accuracy.
        :param integration_accuracy_rel: Relative integration accuracy.
        :param sim_time: How long to simulate for.
        :param alias_spikes: Whether to alias spike times to the numerical integration grid. `False` means that precise integration will be used for spike times whenever possible. `True` means that after taking a timestep :math:`dt` and arriving at :math:`t`, spikes from :math:`\langle t - dt, t]` will only be processed at time :math:`t`.
        """
        self.alias_spikes = alias_spikes
        self.max_step_size = max_step_size
        self.integration_accuracy_abs = integration_accuracy_abs
        self.integration_accuracy_rel = integration_accuracy_rel
        self.sim_time = sim_time
        self._system_of_shapes = system_of_shapes
        self.symbolic_jacobian_ = self._system_of_shapes.get_jacobian_matrix()
        self.shapes = shapes
        self.system_of_shapes = system_of_shapes
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        self.parameters = {k: sympy.parsing.sympy_parser.parse_expr(v, global_dict=Shape._sympy_globals).n() for k, v in self.parameters.items()}
        self._locals = self.parameters.copy()
        if stimuli is None:
            self._stimuli = []
        else:
            self._stimuli = stimuli
        self.random_seed = random_seed

        self.analytic_solver_dict = analytic_solver_dict
        if not self.analytic_solver_dict is None:
            if not "parameters" in self.analytic_solver_dict.keys():
                self.analytic_solver_dict["parameters"] = {}
            self.analytic_solver_dict["parameters"].update(self.parameters)
        self.analytic_integrator = None


    @property
    def random_seed(self):
        return self._random_seed


    @random_seed.setter
    def random_seed(self, value):
        assert type(value) is int
        assert value >= 0
        self._random_seed = value


    def check_stiffness(self, raise_errors=False):
        r"""
        Perform stiffness testing: use implicit and explicit solvers to simulate the dynamical system, then decide which is the better solver to use.

        For details, see https://ode-toolbox.readthedocs.io/en/latest/index.html#numeric-solver-selection-criteria

        :return: Either :python:`"implicit"`, :python:`"explicit"` or :python:`"warning"`.
        :rtype: str
        """
        assert PYGSL_AVAILABLE

        try:
            step_min_exp, step_average_exp, runtime_exp = self._evaluate_integrator(odeiv.step_rk4, raise_errors=raise_errors)
            step_min_imp, step_average_imp, runtime_imp = self._evaluate_integrator(odeiv.step_bsimp, raise_errors=raise_errors)
        except ParametersIncompleteException:
            logging.warning("Stiffness test not possible because numerical values were not specified for all parameters.")
            return None

        # logging.info("runtime (imp:exp): %f:%f" % (runtime_imp, runtime_exp))

        return self._draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp)


    def _evaluate_integrator(self, integrator, h_min_lower_bound=1E-12, raise_errors=True, debug=True):
        r"""
        This function computes the average step size and the minimal step size that a given integration method from GSL uses to evolve a certain system of ODEs during a certain simulation time, integration method from GSL and spike train for a given maximal stepsize.

        This function will reset the numpy random seed.

        :param integrator: A method from the GSL library for evolving ODEs, e.g. :python:`odeiv.step_rk4`.
        :param h_min_lower_bound: The minimum acceptable step size. Integration will terminate with an error if this step size is reached.
        :param raise_errors: Stop and raise exception when error occurs, or try to continue.

        :return h_min: Minimum recommended step size.
        :return h_avg: Average recommended step size.
        :return runtime: Wall clock runtime.
        """
        assert PYGSL_AVAILABLE

        np.random.seed(self.random_seed)

        spike_times = SpikeGenerator.spike_times_from_json(self._stimuli, self.sim_time)


        #
        #  initialise and run mixed integrator
        #

        logging.info("Simulating for " + str(self.sim_time) + " with max_step_size = " + str(self.max_step_size))

        mixed_integrator = MixedIntegrator(integrator,
                                           self.system_of_shapes,
                                           self.shapes,
                                           analytic_solver_dict=self.analytic_solver_dict,
                                           parameters=self.parameters,
                                           spike_times=spike_times,
                                           random_seed=self.random_seed,
                                           max_step_size=self.max_step_size,
                                           integration_accuracy_abs=self.integration_accuracy_abs,
                                           integration_accuracy_rel=self.integration_accuracy_rel,
                                           sim_time=self.sim_time,
                                           alias_spikes=self.alias_spikes)
        h_min, h_avg, runtime = (lambda x: x[:3])(mixed_integrator.integrate_ode(h_min_lower_bound=h_min_lower_bound, raise_errors=raise_errors, debug=debug))

        logging.info("For integrator = " + str(integrator) + ": h_min = " + str(h_min) + ", h_avg = " + str(h_avg) + ", runtime = " + str(runtime))

        return h_min, h_avg, runtime


    def _draw_decision(self, step_min_imp, step_min_exp, step_average_imp, step_average_exp, machine_precision_dist_ratio=10, avg_step_size_ratio=6):
        r"""
        Decide which is the best integrator to use.

        For details, see https://ode-toolbox.readthedocs.io/en/latest/index.html#numeric-solver-selection-criteria

        :param step_min_imp: Minimum recommended step size returned for implicit solver.
        :param step_min_exp: Minimum recommended step size returned for explicit solver.
        :param step_average_imp: Average recommended step size returned for implicit solver.
        :param step_average_exp: Average recommended step size returned for explicit solver.
        """
        machine_precision = np.finfo(float).eps

        if step_min_imp > machine_precision_dist_ratio * machine_precision and step_min_exp < machine_precision_dist_ratio * machine_precision:
            return "implicit"
        elif step_min_imp < machine_precision_dist_ratio * machine_precision and step_min_exp > machine_precision_dist_ratio * machine_precision:
            return "explicit"
        elif step_min_imp < machine_precision_dist_ratio * machine_precision and step_min_exp < machine_precision_dist_ratio * machine_precision:
            return "warning"

        if step_average_imp > avg_step_size_ratio * step_average_exp:
            return "implicit"
        else:
            return "explicit"
