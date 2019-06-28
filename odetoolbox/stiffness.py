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

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    STIFFNESS_DEBUG_PLOT = True
except:
    STIFFNESS_DEBUG_PLOT = False

import sympy
import sympy.utilities.autowrap
import time

try:
    import pygsl.odeiv as odeiv
except ImportError as ie:
    print("Warning: PyGSL is not available. The stiffness test will be skipped.")
    print("Warning: " + str(ie), end="\n\n\n")
    raise


class StiffnessTester(object):

    def __init__(self, system_of_shapes, shapes, analytic_solver_dict=None, parameters={}, stimuli=[], random_seed=123, max_step_size=np.inf, integration_accuracy=1E-3, sim_time=100., alias_spikes=False):
        self.alias_spikes = alias_spikes
        self.max_step_size = max_step_size
        self.integration_accuracy = integration_accuracy
        self.sim_time = sim_time
        self.math_module_funcs = { k : v for k, v in getmembers(math) if not k[0] == "_"}
        self._system_of_shapes = system_of_shapes
        self.symbolic_jacobian_ = self._system_of_shapes.get_jacobian_matrix()
        self._shapes = shapes
        self._parameters = parameters
        self._parameters = { k : sympy.parsing.sympy_parser.parse_expr(v, global_dict=Shape._sympy_globals).n() for k, v in self._parameters.items() }
        self._locals = self._parameters.copy()
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
        self._update_expr_wrapped = {}
        self.all_variable_symbols = list(self._system_of_shapes.x_)
        if not self.analytic_solver_dict is None:
            self.all_variable_symbols += self.analytic_solver_dict["state_variables"]
        self.all_variable_symbols = [ sympy.Symbol(str(sym)) for sym in self.all_variable_symbols ]
        for sym, expr in self._update_expr.items():
            self._update_expr_wrapped[sym] = sympy.utilities.autowrap.autowrap(expr.subs(self._locals), args=self.all_variable_symbols, backend="cython")
        self.symbolic_jacobian_wrapped = np.empty(self.symbolic_jacobian_.shape, dtype=np.object)
        for i in range(self.symbolic_jacobian_.shape[0]):
            for j in range(self.symbolic_jacobian_.shape[1]):
                self.symbolic_jacobian_wrapped[i, j] = sympy.utilities.autowrap.autowrap(self.symbolic_jacobian_[i, j].subs(self._locals), args=self.all_variable_symbols, backend="cython")
        #self._update_expr = { sym : sympy.parsing.sympy_parser.parse_expr(expr, global_dict=Shape._sympy_globals) for sym, expr in self._system_of_shapes.generate_numeric_solver()["update_expressions"].items() }


    @property
    def random_seed(self):
        return self._random_seed


    @random_seed.setter
    def random_seed(self, value):
        assert type(value) is int
        assert value >= 0
        self._random_seed = value


    def check_stiffness(self, raise_errors=False):
        """Perform stiffness testing.

        The idea is not to compare if the given implicit method or the explicit method is better suited for this small simulation, for instance by comparing runtimes, but to instead check for tendencies of stiffness. If we find that the average step size of the implicit evolution method is a lot larger than the average step size of the explicit method, then this ODE system could be stiff. Especially if it become significantly more stiff when a different step size is used or when parameters are changed, an implicit evolution scheme could become increasingly important.

        It is important to note here, that this analysis depends significantly on the parameters that are assigned for an ODE system. If these are changed significantly in magnitude, the result of the analysis can also change significantly.
        """

        step_min_exp, step_average_exp, runtime_exp = self.evaluate_integrator(odeiv.step_rk4, raise_errors=raise_errors)
        step_min_imp, step_average_imp, runtime_imp = self.evaluate_integrator(odeiv.step_bsimp, raise_errors=raise_errors)

        #print("runtime (imp:exp): %f:%f" % (runtime_imp, runtime_exp))

        return self.draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp)


    def evaluate_integrator(self, integrator, h_min_lower_bound=5E-9, sym_receiving_spikes=[], raise_errors=True, debug=True):
        """
        This function computes the average step size and the minimal step size that a given integration method from GSL uses to evolve a certain system of ODEs during a certain simulation time, integration method from GSL and spike train for a given maximal stepsize.

        This function will reset the numpy random seed.

        :param max_step_size: The maximal stepsize for one evolution step in miliseconds
        :param integrator: A method from the GSL library for evolving ODEs, e.g. `odeiv.step_rk4`
        :param y: The 'state variables' in f(y)=y'
        :return: Average and minimal step size.
        """

        h_min = np.inf
        #simulation_slices = int(round(sim_time / max_step_size))

        np.random.seed(self.random_seed)

        self.spike_times = SpikeGenerator.spike_times_from_json(self._stimuli, self.sim_time)


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
        _initial_values = [ float(self._system_of_shapes.get_initial_value(str(sym)).evalf(subs=self._parameters)) for sym in self._system_of_shapes.x_ ]
        y = _initial_values.copy()

        if debug:
            y_log = [y]
            t_log = [0.]
            h_log = []

        gsl_stepper = integrator(len(y), self.step, self.numerical_jacobian)
        control = odeiv.control_y_new(gsl_stepper, self.integration_accuracy, self.integration_accuracy)
        evolve = odeiv.evolve(gsl_stepper, control, len(y))


        #
        #    grab starting wall clock time
        #

        time_start = time.time()


        #
        #    main loop
        #

        h_sum = 0.
        n_timesteps_taken = 0
        t = 0.
        idx_next_spike = 0
        while t < self.sim_time:

            if self.alias_spikes:

                #
                #    simulate by one timestep
                #

                t_target = t + self.max_step_size

            else:

                #
                #    simulate until the time of the next upcoming spike
                #

                if idx_next_spike >= len(self.all_spike_times):
                    t_target = self.sim_time
                    syms_next_spike = []
                else:
                    t_target = self.all_spike_times[idx_next_spike]
                    syms_next_spike = self.all_spike_times_sym[idx_next_spike]

                idx_next_spike += 1		# "queue" the next upcoming spike for the next iteration of the while loop

            while t < t_target:
                t_target_requested = min(t + self.max_step_size, t_target)
                h_requested = t_target_requested - t
                try:
                    if not self.analytic_integrator is None:
                        self.analytic_integrator.disable_cache_update()

                    t, h_suggested, y = evolve.apply(t, t_target_requested, h_requested, y)      # evolve.apply parameters: start time, end time, initial step size, start vector
                except FloatingPointError as e:
                    msg = "Failure of numerical integrator (method: %s) at t=%.2f with requested timestep = %.2f (y = %s)" % (gsl_stepper.name(), t, h_requested, y)
                    raise FloatingPointError(msg)

                if not self.analytic_integrator is None:
                    self.analytic_integrator.enable_cache_update()
                    self.analytic_integrator.get_value(t)

                if debug:
                    t_log.append(t)
                    h_log.append(h_suggested)
                    y_log.append(y)

                if h_suggested < h_requested:     # ignore small requested step sizes; look only at actually obtained step sizes
                    h_min = min(h_min, h_suggested)

                if h_min < h_min_lower_bound:
                    estr = "Integration step below %.e (s=%.f). Please check your ODE." % (h_min_lower_bound, h_min)
                    if raise_errors:
                        raise Exception(estr)
                    else:
                        print(estr)

                h_sum += h_suggested
                n_timesteps_taken += 1


                #
                #    enforce upper and lower thresholds
                #

                for shape in self._shapes:
                    if not shape.upper_bound is None:
                        idx = [str(sym) for sym in list(self._system_of_shapes.x_)].index(str(shape.symbol))
                        upper_bound_numeric = float(shape.upper_bound.evalf(subs=self._locals))
                        if y[idx] > upper_bound_numeric:
                            y[idx] = _initial_values[idx]


            #
            #	evaluate to numeric values for the ODEs that are solved analytically
            #

            self._locals.update({ str(sym) : y[i] for i, sym in enumerate(self._system_of_shapes.x_) })

            if not self.analytic_integrator is None:
                self._locals.update(self.analytic_integrator.get_value(t))


            #
            #   apply the spikes, i.e. add the "initial values" to the system dynamical state vector
            #

            if self.alias_spikes:
                t_next_spike = self.all_spike_times[idx_next_spike]
                while t_next_spike <= t:
                    syms_next_spike = self.all_spike_times_sym[idx_next_spike]
                    for sym in syms_next_spike:
                        if str(sym) in [str(sym_) for sym_ in self._system_of_shapes.x_]:
                            idx = [str(sym) for sym in list(self._system_of_shapes.x_)].index(sym)
                            y[idx] += float(self._system_of_shapes.get_initial_value(str(sym)).evalf(subs=self._locals))
                    idx_next_spike += 1
                    t_next_spike = self.all_spike_times[idx_next_spike]
            else:
                for sym in syms_next_spike:
                    if str(sym) in [str(sym_) for sym_ in self._system_of_shapes.x_]:
                        idx = [str(sym) for sym in list(self._system_of_shapes.x_)].index(sym)
                        y[idx] += float(self._system_of_shapes.get_initial_value(str(sym)).evalf(subs=self._locals))

        h_avg = h_sum / n_timesteps_taken
        runtime = time.time() - time_start

        if debug:
            t_log = np.array(t_log)
            h_log = np.array(h_log)
            y_log = np.array(y_log)
            if not self.analytic_integrator is None:
                analytic_syms = self.analytic_integrator.get_value(t).keys()
                analytic_dim = len(analytic_syms)
                analytic_y_log = { sym : [] for sym in analytic_syms }
                for t in t_log:
                    for sym in analytic_syms:
                        val_dict = self.analytic_integrator.get_value(t)
                        analytic_y_log[sym].append(val_dict[sym])
            else:
                analytic_syms = []
                analytic_dim = 0

            idx_to_label = {}
            for sym, spike_density in self.spike_times.items():
                syms = [str(sym) for sym in list(self._system_of_shapes.x_)]
                if sym.replace("'", "__d") in syms:
                    idx = syms.index(sym.replace("'", "__d"))
                    idx_to_label[idx] = str(sym)
            for i, sym in enumerate(self._system_of_shapes.x_):
                idx_to_label[i] = sym

            if STIFFNESS_DEBUG_PLOT:
                fig, ax = plt.subplots(y_log.shape[1] + analytic_dim + 1, sharex=True)
                for i in range(y_log.shape[1]):
                    sym = idx_to_label[i]
                    ax[i].plot(t_log, y_log[:, i], label=sym, marker="o", color="blue")
                    if sym in self.spike_times.keys():
                        for t_sp in self.spike_times[sym]:
                            ax[i].plot((t_sp, t_sp), ax[i].get_ylim(), marker="o", color="grey", alpha=.7, linewidth=2.)
                for i, sym in enumerate(analytic_syms):
                    ax[i + y_log.shape[1]].plot(t_log, analytic_y_log[sym], label=str(sym), marker="o", color="chartreuse")

                ax[-1].semilogy(t_log[1:], h_log, linewidth=2, color="grey", marker="o", alpha=.7)
                ax[-1].set_ylabel("suggested dt [s]")

                for _ax in ax:
                    _ax.legend()
                    _ax.grid(True)
                    _ax.set_xlim(0., np.amax(t_log))

                ax[-1].set_xlabel("Time [s]")
                fig.suptitle(str(integrator))
                #plt.show()
                fn = "/tmp/remotefs2/stiffness_test_" + str(integrator) + ".png"
                print("Saving to " + fn)
                plt.savefig(fn, dpi=600)

        print("For integrator = " + str(integrator) + ": h_min = " + str(h_min) + ", h_avg = " + str(h_avg) + ", runtime = " + str(runtime))

        return h_min, h_avg, runtime


    def draw_decision(self, step_min_imp, step_min_exp, step_average_imp, step_average_exp, avg_step_size_ratio=6):
        """Decide which is the best integrator to use for a certain system of ODEs

        1. If the minimal step size is close to machine precision for one of the methods but not for the other, this suggest that the other is more stable and should be used instead.

        2. If the ODE system is stiff the average step size of the implicit method tends to be larger. This indicates that the ODE system is possibly stiff (and that it could be even more stiff for minor changes in stepsize and parameters). 

        :param step_min_imp: data measured during solving
        :param step_min_exp: data measured during solving
        :param step_average_imp: data measured during solving
        :param step_average_exp: data measured during solving
        """

        machine_precision = np.finfo(float).eps

        if step_min_imp > 10. * machine_precision and step_min_exp < 10. * machine_precision:
            return "implicit"
        elif step_min_imp < 10. * machine_precision and step_min_exp > 10. * machine_precision:
            return "explicit"
        elif step_min_imp < 10. * machine_precision and step_min_exp < 10. * machine_precision:
            return "warning"

        if step_average_imp > avg_step_size_ratio * step_average_exp:
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
        print("begin evaulating jacobian")
        dimension = len(y)
        dfdy = np.zeros((dimension, dimension), np.float)
        dfdt = np.zeros((dimension,))

        self._locals.update({ str(sym) : y[i] for i, sym in enumerate(self._system_of_shapes.x_) })

        if not self.analytic_integrator is None:
            self._locals.update(self.analytic_integrator.get_value(t))

        # y holds the state of all the symbols in the numeric part of the system; add those for the analytic part
        y = [ self._locals[str(sym)] for sym in self.all_variable_symbols ]

        # evaluate every entry of the `jacobian_matrix` and store the
        # result in the corresponding entry of the `dfdy`
        for row in range(0, dimension):
            for col in range(0, dimension):
                dfdy[row, col] = self.symbolic_jacobian_wrapped[row, col](*y)
                #dfdy[row, col] = float(self.symbolic_jacobian_[row, col].evalf(subs=self._locals))
        print("end evaulating jacobian")
        return dfdy, dfdt


    def step(self, t, y, params):
        """Callback function to compute an integration step.

        :param t: current time in the step (from 0 to step_size)
        :param y: the current state vector of the ODE system
        :param params: Prescribed GSL parameters (not used here).

        :return: Updated state vector
        """

        print("begin step, t = " + str(t))
        self._locals.update({ str(sym) : y[i] for i, sym in enumerate(self._system_of_shapes.x_) })

        #
        #   update state of analytically solved variables to time `t`
        #

        if not self.analytic_integrator is None:
            print("\tbegin analytic get")
            self._locals.update(self.analytic_integrator.get_value(t))
            print("\tend analytic get")

        # y holds the state of all the symbols in the numeric part of the system; add those for the analytic part
        y = [ self._locals[str(sym)] for sym in self.all_variable_symbols ]

        try:
            #return [ float(self._update_expr[str(sym)].evalf(subs=self._locals)) for sym in self._system_of_shapes.x_ ]
            _ret = [ self._update_expr_wrapped[str(sym)](*y) for sym in self._system_of_shapes.x_ ]
        except Exception as e:
            print("E==>", type(e).__name__ + ": " + str(e))
            print("     Local parameters at time of failure:")
            for k,v in self._locals.items():
                print("    ", k, "=", v)
            raise
        print("end step")
        return _ret
