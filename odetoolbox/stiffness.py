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

from __future__ import print_function

from inspect import getmembers
import math
import random
import numpy as np
import numpy.random
from .shapes import Shape
from .analytic_integrator import AnalyticIntegrator
from .spike_generator import SpikeGenerator

# Make NumPy warnings errors. Without this, we can't catch overflow errors that can occur in the step() function, which might indicate a problem with the ODE, the grid resolution or the stiffness testing framework itself.
np.seterr(over='raise')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import time

import sympy
#from sympy.parsing.sympy_parser import parse_expr

try:
    import pygsl.odeiv as odeiv
except ImportError as ie:
    print("Warning: PyGSL is not available. The stiffness test will be skipped.")
    print("Warning: " + str(ie), end="\n\n\n")
    raise


class StiffnessTester(object):

    def __init__(self, system_of_shapes, shapes, analytic_solver_dict=None, parameters={}, stimuli=[], random_seed=123):
        self.math_module_funcs = { k : v for k, v in getmembers(math) if not k[0] == "_"}
        self._system_of_shapes = system_of_shapes
        self.symbolic_jacobian_ = self._system_of_shapes.get_jacobian_matrix()
        self._shapes = shapes
        self._parameters = parameters
        self._parameters = { k : sympy.parsing.sympy_parser.parse_expr(v, global_dict=Shape._sympy_globals).n() for k, v in self._parameters.items() }
        self._stimuli = stimuli

        self.random_seed = random_seed

        self.analytic_solver_dict = analytic_solver_dict
        if not self.analytic_solver_dict is None:
            if not "parameters" in self.analytic_solver_dict.keys():
                self.analytic_solver_dict["parameters"] = {}
            self.analytic_solver_dict["parameters"].update(self._parameters)
        self.analytic_integrator = None
        #self.initial_values = { sym : str(self.get_initial_value(sym)) for sym in self._system_of_shapes.x_ }
        self._update_expr = self._system_of_shapes.generate_numeric_solver()["update_expressions"].copy()
        #self._update_expr = { sym : sympy.parsing.sympy_parser.parse_expr(expr, global_dict=Shape._sympy_globals) for sym, expr in self._system_of_shapes.generate_numeric_solver()["update_expressions"].items() }

    @property
    def random_seed(self):
        return self._random_seed


    @random_seed.setter
    def random_seed(self, value):
        assert type(value) is int
        assert value >= 0
        self._random_seed = value


    def check_stiffness(self, sim_resolution=.1, sim_time=1., accuracy=1E-3):
        """Perform stiffness testing.

        The idea is not to compare if the given implicit method or the explicit method is better suited for this small simulation, for instance by comparing runtimes, but to instead check for tendencies of stiffness. If we find that the average step size of the implicit evolution method is a lot larger than the average step size of the explicit method, then this ODE system could be stiff. Especially if it become significantly more stiff when a different step size is used or when parameters are changed, an implicit evolution scheme could become increasingly important.

        It is important to note here, that this analysis depends significantly on the parameters that are assigned for an ODE system. If these are changed significantly in magnitude, the result of the analysis can also change significantly.
        """

        step_min_exp, step_average_exp, runtime_exp = self.evaluate_integrator_exp(sim_resolution, accuracy, sim_time)
        step_min_imp, step_average_imp, runtime_imp = self.evaluate_integrator_imp(sim_resolution, accuracy, sim_time)

        #print("runtime (imp:exp): %f:%f" % (runtime_imp, runtime_exp))

        return self.draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp)


    def evaluate_integrator_imp(self, sim_resolution, accuracy, sim_time, raise_errors=True):

        integrator = odeiv.step_bsimp     # Bulirsh-Stoer
        return self.evaluate_integrator(sim_resolution, integrator, accuracy, sim_time, raise_errors=raise_errors)


    def evaluate_integrator_exp(self, sim_resolution, accuracy, sim_time, raise_errors=True):

        integrator = odeiv.step_rk4       # explicit 4th order Runge-Kutta
        return self.evaluate_integrator(sim_resolution, integrator, accuracy, sim_time, raise_errors=raise_errors)


    def evaluate_integrator(self, h, integrator, accuracy, sim_time, s_min_lower_bound=5E-9, sym_receiving_spikes=[], raise_errors=True, max_step_size=2E-3, debug=True):
        """
        This function computes the average step size and the minimal step size that a given integration method from GSL uses to evolve a certain system of ODEs during a certain simulation time, integration method from GSL and spike train for a given maximal stepsize.

        This function will reset the numpy random seed.

        :param h: The maximal stepsize for one evolution step in miliseconds
        :param integrator: A method from the GSL library for evolving ODEs, e.g. `odeiv.step_rk4`
        :param y: The 'state variables' in f(y)=y'
        :return: Average and minimal step size.
        """

        s_min = h  # initialised at upper bound: minimal step size <= the maximal stepsize h
        #simulation_slices = int(round(sim_time / h))

        np.random.seed(self.random_seed)

        self.spike_times = SpikeGenerator.spike_times_from_json(self._stimuli, sim_time)


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


        #
        #  initialise analytic integrator
        #

        if not self.analytic_solver_dict is None:
            self.analytic_integrator = AnalyticIntegrator(self.analytic_solver_dict, self.spike_times)
            #analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)

        #N = len(self._system_of_shapes.x_)
        _initial_values = [ float(self._system_of_shapes.get_initial_value(str(sym)).subs(self._parameters).evalf()) for sym in self._system_of_shapes.x_ ]
        y = _initial_values.copy()

        if debug:
            y_log = [y]
            t_log = [0.]

        gsl_stepper = integrator(len(y), self.step, self.numerical_jacobian)
        control = odeiv.control_y_new(gsl_stepper, accuracy, accuracy)
        evolve = odeiv.evolve(gsl_stepper, control, len(y))


        #
        #    grab starting wall clock time
        #

        time_start = time.time()


        #
        #    main loop
        #

        t = 0.
        idx_next_spike = 0
        while t < sim_time:

            #
            # find the time of the next upcoming spike
            #

            """all_spike_times = []
            for sym in syms:
                all_spike_times.extend(spike_times[sym])
            all_spike_times = np.sort(all_spike_times)
            all_spike_times = [t_sp for t_sp in all_spike_times if t_sp > t]
            t_next_spike = all_spike_times[0]"""

            print("Top loop: t = " + str(t))


            #
            # simulate until the time of the next upcoming spike
            #

            if idx_next_spike >= len(self.all_spike_times):
                t_next_spike = sim_time
            else:
                t_next_spike = self.all_spike_times[idx_next_spike]

            print("\tsimulating inner till spike or end t = " + str(t_next_spike))

            while t < t_next_spike:
                try:
                    # h_ is NOT the reached step size but the suggested next step size!
                    t, h_, y = evolve.apply(t, min(t + max_step_size, t_next_spike), h, y)
                except Exception as e:
                    print("     ===> Failure of %s at t=%.2f with h=%.2f (y=%s)" % (gsl_stepper.name(), t, h, y))
                    if raise_errors:
                        raise e

                if debug:
                    t_log.append(t)
                    y_log.append(y)


                """#step_counter += 1
                s_min_old = s_min
                s_min = min(s_min, t - t_old)
                if s_min < s_min_lower_bound:
                    estr = "Integration step below %.e (s=%.f). Please check your ODE." % (s_min_lower_bound, s_min)
                    if raise_errors:
                        raise Exception(estr)
                    else:
                        print(estr)"""

            idx_next_spike += 1


            """if counter_while_loop > 1:
                step_counter -= 1
                sum_last_steps += t_new - t_old
                # it is possible that the last step in a simulation_slot is very small, as it is simply
                # the length of the remaining slot. Therefore we don't take the last step into account
                s_min = s_min_old"""


            #
            #	evaluate to numeric values for the ODEs that are solved analytically
            #

            _locals = self._parameters.copy()
            _locals.update({ str(sym) : y[i] for i, sym in enumerate(self._system_of_shapes.x_) })

            if not self.analytic_integrator is None:
                _locals.update(self.analytic_integrator.get_value(t))


            #
            #	enforce upper and lower thresholds
            #

            for shape in self._shapes:
                if not shape.upper_bound is None:
                    idx = [str(sym) for sym in list(self._system_of_shapes.x_)].index(str(shape.symbol))
                    upper_bound_numeric = float(shape.upper_bound.subs(_locals).evalf())
                    if y[idx] > upper_bound_numeric:
                        y[idx] = _initial_values[idx]


            #
            #   apply the spikes, i.e. add the "initial values" to the system dynamical state vector
            #

            for sym, spike_density in self.spike_times.items():
                if sym in list(self._system_of_shapes.x_):
                    idx = [str(sym) for sym in list(self._system_of_shapes.x_)].index(sym)
                    y[idx] += float(self._system_of_shapes.get_initial_value(str(sym)).subs(_locals).evalf())

        #step_average = (t - sum_last_steps) / step_counter

        #runtime = time.time() - time_start

        if debug:
            t_log = np.array(t_log)
            y_log = np.array(y_log)
            analytic_syms = self.analytic_integrator.get_value(t).keys()
            analytic_dim = len(analytic_syms)
            analytic_y_log = { sym : [] for sym in analytic_syms }
            for t in t_log:
                for sym in analytic_syms:
                    val_dict = self.analytic_integrator.get_value(t)
                    analytic_y_log[sym].append(val_dict[sym])

            idx_to_label = {}
            for sym, spike_density in self.spike_times.items():
                syms = [str(sym) for sym in list(self._system_of_shapes.x_)]
                if sym.replace("'", "__d") in syms:
                    idx = syms.index(sym)
                    idx_to_label[idx] = str(sym)
            for i, sym in enumerate(self._system_of_shapes.x_):
                idx_to_label[i] = sym

            fig, ax = plt.subplots(y_log.shape[1] + len(analytic_syms), sharex=True)
            for i in range(y_log.shape[1]):
                ax[i].plot(t_log, y_log[:, i], label=idx_to_label[i], marker="o", color="blue")
            for i, sym in enumerate(analytic_syms):
                ax[i + y_log.shape[1]].plot(t_log, analytic_y_log[sym], label=str(sym), marker="o", color="chartreuse")

            for _ax in ax:
                _ax.legend()
                _ax.grid(True)
                _ax.set_xlim(0., np.amax(t_log))

            ax[-1].set_xlabel("Time [ms]")
            fig.suptitle(str(integrator))

            #plt.show()
            fn = "/tmp/remotefs2/stiffness_test_" + str(integrator) + ".png"
            print("Saving to " + fn)
            plt.savefig(fn, dpi=600)

        return s_min_old, step_average, runtime


    def draw_decision(self, step_min_imp, step_min_exp, step_average_imp, step_average_exp):
        """
        This function takes the minimal and average step size of the implicit and explicit evolution method.
        The idea is 1. that if the ODE system is stiff the average step size of the implicit method tends to
        be larger. The function checks if it is twice as large. This points to the facht that the ODE system
        is possibly stiff expecially that it could be even more stiff for minor changes in stepzise and and
        parameters. Further if the minimal step size is close to machine precision for one of the methods but
        not for the other, this suggest that the other is more stable and should be used instead.
        :param step_min_imp: data measured during solving
        :param step_min_exp: data measured during solving
        :param step_average_imp: data measured during solving
        :param step_average_exp: data measured during solving
        """
        # check minimal step lengths as used by GSL integration method

        machine_precision = np.finfo(float).eps

        if step_min_imp > 10. * machine_precision and step_min_exp < 10. * machine_precision:
            return "implicit"
        elif step_min_imp < 10. * machine_precision and step_min_exp > 10. * machine_precision:
            return "explicit"
        elif step_min_imp < 10. * machine_precision and step_min_exp < 10. * machine_precision:
            return "warning"
        elif step_min_imp > 10. * machine_precision and step_min_exp > 10. * machine_precision:
            if step_average_imp > 6*step_average_exp:
                return "implicit"
            else:
                return "explicit"



    def numerical_jacobian(self, t, y, params):
        """Callback function that compute the jacobian matrix for the current
        state vector `y`.

        :param t: current time in the step (from 0 to step_size)
        :param y: the current state vector of the ODE system
        :param params: Prescribed GSL parameters (not used here).

        :return: dfdy that contains the jacobian matrix with respect
        to y. `dfdt` is not computed and set to zero matrix now.

        """
        dimension = len(y)
        dfdy = np.zeros((dimension, dimension), np.float)
        dfdt = np.zeros((dimension,))

        _locals = self._parameters.copy()
        _locals.update({ str(sym) : y[i] for i, sym in enumerate(self._system_of_shapes.x_) })

        if not self.analytic_integrator is None:
            _locals.update(self.analytic_integrator.get_value(t))

        # evaluate every entry of the `jacobian_matrix` and store the
        # result in the corresponding entry of the `dfdy`
        for row in range(0, dimension):
            for col in range(0, dimension):
                dfdy[row, col] = float(self.symbolic_jacobian_[row, col].subs(_locals).evalf())

        return dfdy, dfdt


    def step(self, t, y, params):
        """Callback function to compute an integration step.

        :param t: current time in the step (from 0 to step_size)
        :param y: the current state vector of the ODE system
        :param params: Prescribed GSL parameters (not used here).

        :return: Updated state vector
        """

        _locals = self._parameters.copy()
        _locals.update({ str(sym) : y[i] for i, sym in enumerate(self._system_of_shapes.x_) })

        #
        #   update state of analytically solved variables to time `t`
        #

        if not self.analytic_integrator is None:
            # XXX: TODO: clamp analytic solution to bounds (if they exist)
            _locals.update(self.analytic_integrator.get_value(t))

        try:
           return [ float(self._update_expr[str(sym)].subs(_locals).evalf()) for sym in self._system_of_shapes.x_ ]
        except Exception as e:
            print("E==>", type(e).__name__ + ": " + str(e))
            print("     Local parameters at time of failure:")
            for k,v in _locals.items():
                print("    ", k, "=", v)
            raise
