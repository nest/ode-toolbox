#
# mixed_integrator.py
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

from typing import Optional

import logging
import numpy as np
import numpy.random
import os
import sympy
import sympy.utilities.autowrap
from sympy.utilities.autowrap import CodeGenArgumentListError
import time


from .analytic_integrator import AnalyticIntegrator
from .config import Config
from .integrator import Integrator
from .plot_helper import import_matplotlib
from .shapes import Shape
from .sympy_helpers import _is_sympy_type

try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError as ie:
    logging.warning("PyGSL is not available. The stiffness test will be skipped.")
    logging.warning("Error when importing: " + str(ie))
    PYGSL_AVAILABLE = False


class ParametersIncompleteException(Exception):
    """
    Thrown in case not all parameters are assigned a numerical value before integration was attempted.
    """
    pass


class MixedIntegrator(Integrator):
    r"""
    Mixed numeric+analytic integrator. Supply with a result from ODE-toolbox analysis; calculates numeric approximation of the solution.
    """

    def __init__(self, numeric_integrator, system_of_shapes, shapes, analytic_solver_dict=None, parameters=None, spike_times=None, random_seed=123, max_step_size=np.inf, integration_accuracy_abs=1E-6, integration_accuracy_rel=1E-6, sim_time=1., alias_spikes=False, debug_plot_dir: Optional[str] = None):
        r"""
        :param numeric_integrator: A method from the GSL library for evolving ODEs, e.g. :python:`odeiv.step_rk4`
        :param system_of_shapes: Dynamical system to solve.
        :param shapes: List of shapes in the dynamical system.
        :param analytic_solver_dict: Analytic solver dictionary from ODE-toolbox analysis result.
        :param parameters: Dictionary mapping parameter name (as string) to value expression.
        :param spike_times: For each variable, used as a key, the list of times at which a spike occurs.
        :param random_seed: Random number generator seed.
        :param max_step_size: The maximum step size taken by the integrator.
        :param integration_accuracy_abs: Absolute integration accuracy.
        :param integration_accuracy_rel: Relative integration accuracy.
        :param sim_time: How long to simulate for.
        :param alias_spikes: Whether to alias spike times to the numerical integration grid. :python:`False` means that precise integration will be used for spike times whenever possible. :python:`True` means that after taking a timestep :math:`dt` and arriving at :math:`t`, spikes from :math:`\langle t - dt, t]` will only be processed at time :math:`t`.
        :param debug_plot_dir: If given, enable debug plotting to this directory. If enabled, matplotlib is imported and used for plotting.
        """
        super(MixedIntegrator, self).__init__()

        assert PYGSL_AVAILABLE

        self._debug_plot_dir = debug_plot_dir
        self.numeric_integrator = numeric_integrator
        self.alias_spikes = alias_spikes
        self.max_step_size = max_step_size
        self.integration_accuracy_abs = integration_accuracy_abs
        self.integration_accuracy_rel = integration_accuracy_rel
        self.sim_time = sim_time
        self._system_of_shapes = system_of_shapes
        self.symbolic_jacobian_ = self._system_of_shapes.get_jacobian_matrix()
        self._shapes = shapes
        if parameters is None:
            self._parameters = {}
        else:
            self._parameters = parameters
        self._parameters = {k: sympy.parsing.sympy_parser.parse_expr(v, global_dict=Shape._sympy_globals).n() if not _is_sympy_type(v) else v for k, v in self._parameters.items()}
        self._locals = self._parameters.copy()
        self.random_seed = random_seed

        self.analytic_solver_dict = analytic_solver_dict
        if not self.analytic_solver_dict is None:
            if not "parameters" in self.analytic_solver_dict.keys():
                self.analytic_solver_dict["parameters"] = {}
            self.analytic_solver_dict["parameters"].update(self._parameters)
        self.analytic_integrator = None
        self._update_expr = self._system_of_shapes.generate_numeric_solver()["update_expressions"].copy()
        self._update_expr_wrapped = {}

        self.all_variable_symbols = list(self._system_of_shapes.x_)
        if not self.analytic_solver_dict is None:
            self.all_variable_symbols += self.analytic_solver_dict["state_variables"]
        self.all_variable_symbols = [sympy.Symbol(str(sym).replace("'", Config().differential_order_symbol)) for sym in self.all_variable_symbols]

        for sym, expr in self._update_expr.items():
            try:
                self._update_expr_wrapped[sym] = sympy.utilities.autowrap.autowrap(expr.subs(self._locals),
                                                                                   args=self.all_variable_symbols,
                                                                                   backend="cython",
                                                                                   helpers=Shape._sympy_autowrap_helpers)
            except CodeGenArgumentListError:
                raise ParametersIncompleteException("Integration not possible because numerical values were not specified for all parameters.")
        self.symbolic_jacobian_wrapped = np.empty(self.symbolic_jacobian_.shape, dtype=object)
        for i in range(self.symbolic_jacobian_.shape[0]):
            for j in range(self.symbolic_jacobian_.shape[1]):
                self.symbolic_jacobian_wrapped[i, j] = sympy.utilities.autowrap.autowrap(self.symbolic_jacobian_[i, j].subs(self._locals),
                                                                                         args=self.all_variable_symbols,
                                                                                         backend="cython",
                                                                                         helpers=Shape._sympy_autowrap_helpers)


        #
        #   make a sorted list of all spike times for all symbols
        #

        self.set_spike_times(spike_times)


    def integrate_ode(self, initial_values=None, h_min_lower_bound=5E-9, raise_errors=True, debug=False):
        r"""
        This function computes the average step size and the minimal step size that a given integration method from GSL uses to evolve a certain system of ODEs during a certain simulation time, integration method from GSL and spike train.

        :param initial_values: A dictionary mapping variable names (as strings) to initial value expressions.
        :param h_min_lower_bound: The minimum acceptable step size.
        :param raise_errors: Stop and raise exception when error occurs, or try to continue.
        :param debug: Return extra values useful for debugging.

        :return: Average and minimal step size, and elapsed wall clock time.
        """
        if initial_values is None:
            initial_values = {}

        assert all([isinstance(k, sympy.Symbol) for k in initial_values.keys()]), 'Initial value dictionary keys should be of type sympy.Symbol'

        #
        #   grab stimulus spike times
        #

        all_spike_times, all_spike_times_sym = self.get_sorted_spike_times()


        #
        #  initialise analytic integrator
        #

        if not self.analytic_solver_dict is None:
            analytic_integrator_spike_times = {sym: st for sym, st in self.get_spike_times().items() if str(sym) in self.analytic_solver_dict["state_variables"]}
            self.analytic_integrator = AnalyticIntegrator(self.analytic_solver_dict, analytic_integrator_spike_times)
            analytic_integrator_initial_values = {sym: iv for sym, iv in initial_values.items() if sym in self.analytic_integrator.get_all_variable_symbols()}
            self.analytic_integrator.set_initial_values(analytic_integrator_initial_values)


        #
        #    convert initial value expressions to floats
        #

        for sym in self._system_of_shapes.x_:
            if not sym in initial_values.keys():
                initial_values[sym] = float(self._system_of_shapes.get_initial_value(str(sym)).evalf(subs=self._parameters))

        upper_bound_crossed = False
        y = np.array([initial_values[sym] for sym in self._system_of_shapes.x_])

        if debug:
            y_log = [y]
            t_log = [0.]
            h_log = []

        gsl_stepper = self.numeric_integrator(len(y), self.step, self.numerical_jacobian)
        control = odeiv.control_y_new(gsl_stepper, self.integration_accuracy_abs, self.integration_accuracy_rel)
        evolve = odeiv.evolve(gsl_stepper, control, len(y))


        #
        #    make NumPy warnings errors. Without this, we can't catch overflow errors that can occur in the step() function, which might indicate a problem with the ODE, the grid resolution or the stiffness testing framework itself.
        #

        old_numpy_fp_overflow_error_level = np.geterr()['over']
        np.seterr(over='raise')
        try:
            h_min = np.inf

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

                    if idx_next_spike >= len(all_spike_times):
                        t_target = self.sim_time
                        syms_next_spike = []
                    else:
                        t_target = all_spike_times[idx_next_spike]
                        if t_target >= self.sim_time:
                            t_target = self.sim_time
                            syms_next_spike = []
                        else:
                            syms_next_spike = all_spike_times_sym[idx_next_spike]

                    idx_next_spike += 1        # "queue" the next upcoming spike for the next iteration of the while loop

                while t < t_target:
                    t_target_requested = min(t + self.max_step_size, t_target)
                    h_requested = t_target_requested - t
                    try:
                        if not self.analytic_integrator is None:
                            self.analytic_integrator.disable_cache_update()

                        t, h_suggested, y = evolve.apply(t, t_target_requested, h_requested, y)      # evolve.apply parameters: start time, end time, initial step size, start vector
                    except FloatingPointError:
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
                        logging.warning(estr)
                        if raise_errors:
                            raise Exception(estr)

                    h_sum += h_suggested
                    n_timesteps_taken += 1


                    #
                    #    enforce bounds/thresholds
                    #

                    for shape in self._shapes:
                        if not shape.upper_bound is None:
                            idx = [str(sym) for sym in list(self._system_of_shapes.x_)].index(str(shape.symbol))
                            upper_bound_numeric = float(shape.upper_bound.evalf(subs=self._locals))
                            if y[idx] > upper_bound_numeric:
                                upper_bound_crossed = True
                                y[idx] = initial_values[shape.symbol]


                #
                #    evaluate to numeric values those ODEs that are solved analytically
                #

                self._locals.update({str(sym): y[i] for i, sym in enumerate(self._system_of_shapes.x_)})

                if not self.analytic_integrator is None:
                    self._locals.update(self.analytic_integrator.get_value(t))


                #
                #   apply the spikes, i.e. add the "initial values" to the system dynamical state vector
                #

                if self.alias_spikes:

                    #
                    #    apply all the spikes from <t - dt, t]
                    #

                    if idx_next_spike < len(all_spike_times):
                        t_next_spike = all_spike_times[idx_next_spike]
                    else:
                        t_next_spike = np.inf

                    while t_next_spike <= t:
                        syms_next_spike = all_spike_times_sym[idx_next_spike]
                        for sym in syms_next_spike:
                            if sym in [str(sym_) for sym_ in self._system_of_shapes.x_]:
                                idx = [str(sym_) for sym_ in list(self._system_of_shapes.x_)].index(sym)
                                y[idx] += float(self._system_of_shapes.get_initial_value(sym).evalf(subs=self._locals))
                        idx_next_spike += 1
                        if idx_next_spike < len(all_spike_times):
                            t_next_spike = all_spike_times[idx_next_spike]
                        else:
                            t_next_spike = np.inf
                else:
                    for sym in syms_next_spike:
                        if sym in [str(sym_) for sym_ in self._system_of_shapes.x_]:
                            idx = [str(sym_) for sym_ in list(self._system_of_shapes.x_)].index(sym)
                            y[idx] += float(self._system_of_shapes.get_initial_value(sym).evalf(subs=self._locals))

            h_avg = h_sum / n_timesteps_taken
            runtime = time.time() - time_start

        finally:
            np.seterr(over=old_numpy_fp_overflow_error_level)

        if debug:
            t_log = np.array(t_log)
            h_log = np.array(h_log)
            y_log = np.array(y_log)

            if self._debug_plot_dir:
                self.integrator_debug_plot(t_log, h_log, y_log, dir=self._debug_plot_dir)

        logging.info("For integrator = " + str(self.numeric_integrator) + ": h_min = " + str(h_min) + ", h_avg = " + str(h_avg) + ", runtime = " + str(runtime))

        sym_list = self._system_of_shapes.x_

        if debug:
            return h_min, h_avg, runtime, upper_bound_crossed, t_log, h_log, y_log, sym_list
        else:
            return h_min, h_avg, runtime


    def integrator_debug_plot(self, t_log, h_log, y_log, dir):
        mpl, plt = import_matplotlib()
        assert mpl, "Debug plot was requested for MixedIntegrator, but an exception occurred while importing matplotlib. See the ``debug_plot_dir`` parameter."

        if not self.analytic_integrator is None:
            analytic_syms = self.analytic_integrator.get_value(0.).keys()
            analytic_dim = len(analytic_syms)
            analytic_y_log = {sym: [] for sym in analytic_syms}
            for t in t_log:
                for sym in analytic_syms:
                    val_dict = self.analytic_integrator.get_value(t)
                    analytic_y_log[sym].append(val_dict[sym])
        else:
            analytic_syms = []
            analytic_dim = 0

        idx_to_label = {}
        for sym, spike_density in self.get_spike_times().items():
            syms = [sym for sym in list(self._system_of_shapes.x_)]
            if sym in syms:
                idx = syms.index(sym)
                idx_to_label[idx] = sym

        for i, sym in enumerate(self._system_of_shapes.x_):
            idx_to_label[i] = sym

        fig, ax = plt.subplots(y_log.shape[1] + analytic_dim + 1, sharex=True)

        # adjust the axes slightly towards the right
        for _ax in ax:
            pos1 = _ax.get_position()
            pos2 = [pos1.x0 + 0.05, pos1.y0, pos1.width, pos1.height]
            _ax.set_position(pos2)

        for i in range(y_log.shape[1]):
            sym = idx_to_label[i]
            ax[i].plot(t_log, y_log[:, i], label=sym, marker="o", color="blue")
            if sym in self.get_spike_times().keys():
                for t_sp in self.get_spike_times()[sym]:
                    ax[i].plot((t_sp, t_sp), ax[i].get_ylim(), marker="o", color="grey", alpha=.7, linewidth=2.)
        for i, sym in enumerate(analytic_syms):
            ax[i + y_log.shape[1]].plot(t_log, analytic_y_log[sym], label=str(sym), marker="o", color="chartreuse")

        ax[-1].semilogy(t_log[1:], h_log, linewidth=2, color="grey", marker="o", alpha=.7)
        ax[-1].set_ylabel("suggested dt [s]")

        for _ax in ax:
            if not _ax is ax[-1]:
                _ax.legend()
            _ax.grid(True)
            _ax.set_xlim(0., np.amax(t_log))

        ax[-1].set_xlabel("Time [s]")
        fig.suptitle(str(self.numeric_integrator))
        fn = os.path.join(dir, "mixed_integrator_[run=" + str(hash(self)) + "]_[run=" + str(self.numeric_integrator) + "].png")
        plt.savefig(fn, dpi=600)
        plt.close(fig)


    def numerical_jacobian(self, t, y, params):
        r"""
        Compute the numerical values of the Jacobian matrix at the current time :python:`t` and state :python:`y`.

        If the dynamics of variables :math:`x_1, \ldots, x_N` is defined as :math:`x_i' = f_i`, then row :math:`i` of the Jacobian matrix :math:`\mathbf{J}_i = \left[\begin{matrix}\frac{\partial f_i}{\partial x_0} & \cdots & \frac{\partial f_i}{\partial x_N}\end{matrix}\right]`.

        :param t: Current time.
        :param y: Current state vector of the dynamical system.
        :param params: GSL parameters (not used here).

        :return: Jacobian matrix :math:`\mathbf{J}`.
        """
        dimension = len(y)
        dfdy = np.zeros((dimension, dimension), float)
        dfdt = np.zeros((dimension,))

        self._locals.update({str(sym): y[i] for i, sym in enumerate(self._system_of_shapes.x_)})

        if not self.analytic_integrator is None:
            self._locals.update(self.analytic_integrator.get_value(t))

        y = [self._locals[str(sym)] for sym in self.all_variable_symbols]

        for row in range(0, dimension):
            for col in range(0, dimension):
                dfdy[row, col] = self.symbolic_jacobian_wrapped[row, col](*y)
                # dfdy[row, col] = float(self.symbolic_jacobian_[row, col].evalf(subs=self._locals))	# non-wrapped version

        return dfdy, dfdt


    def step(self, t, y, params):
        r"""
        "Stepping function": compute the (numerical) value of the derivative of :python:`y` over time, at the current time :python:`t` and state :python:`y`.

        :param t: Current time.
        :param y: Current state vector of the dynamical system.
        :param params: GSL parameters (not used here).

        :return: Updated state vector
        """
        self._locals.update({str(sym): y[i] for i, sym in enumerate(self._system_of_shapes.x_)})

        #
        #   update state of analytically solved variables to time `t`
        #

        if not self.analytic_integrator is None:
            self._locals.update(self.analytic_integrator.get_value(t))

        # y holds the state of all the symbols in the numeric part of the system; add those for the analytic part
        y = [self._locals[str(sym)] for sym in self.all_variable_symbols]

        try:
            # return [ float(self._update_expr[str(sym)].evalf(subs=self._locals)) for sym in self._system_of_shapes.x_ ]	# non-wrapped version
            _ret = [self._update_expr_wrapped[str(sym)](*y) for sym in self._system_of_shapes.x_]
        except Exception as e:
            logging.error("E==>", type(e).__name__ + ": " + str(e))
            logging.error("     Local parameters at time of failure:")
            for k, v in self._locals.items():
                logging.error("    ", k, "=", v)
            raise

        return _ret
