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
import numpy
import numpy.random

# Make NumPy warnings errors. Without this, we can't catch overflow
# errors that can occur in the step() function, which might indicate a
# problem with the ODE, the grid resolution or the stiffness testing
# framework itself.
numpy.seterr(all='raise')

import re
import time

import sympy
from sympy.parsing.sympy_parser import parse_expr

try:
    import pygsl.odeiv as odeiv
except ImportError as ie:
    print("Warning: PyGSL is not available. The stiffness test will be skipped.")
    print("Warning: " + str(ie), end="\n\n\n")
    raise


class StiffnessTester(object):

    def __init__(self, indict):
        self.math_module_funcs = {k: v for k, v in getmembers(math) if not k[0] == "_"}
        self._read_input(indict)


    def _read_input(self, indict):
        """Parse the input and store a representation of the ODE system that
        can be propagated by the GSL in the member variables
        `ode_definitions`, `initial_values`, `state_start_values` and
        `thresholds`.

        This function expects a single dictionary with the keys `odes`,
        `parameters` and `shapes` that describe the input to the analysis.

        The exact format of the input entries is described in the file
        `README.md`.

        Parameters
        ----------
        indict: dict
          A dictionary that encodes the ODE system.

        """

        ode_definitions_tmp = {}
        initial_values_tmp = {}
        state_start_values_tmp = {}
        thresholds_tmp = []

        self.parameters = {k:float(v) for k,v in indict["parameters"].items()}

        for shape in indict["shapes"]:
            shape_name = shape["symbol"]
            max_order = len(shape["initial_values"])
            for order in range(0, max_order - 1):
                ode_definitions_tmp[shape_name + "__d" * (order - 1)] = shape_name + "__d" * (order - 1)
                initial_values_tmp[shape_name + "__d" * order] = shape["initial_values"][order]
            ode_definitions_tmp[shape_name + "__d" * (max_order - 1)] = shape["definition"].replace("'", "__d")
            initial_values_tmp[shape_name + "__d" * (max_order - 1)] = shape["initial_values"][(max_order - 1)]

        # we omit initial values for odes since they cannot be connected to buffers
        for ode in indict["odes"]:
            ode_lhs = ode["symbol"]
            max_order = len(ode["initial_values"])
            for order in range(0, len(ode["initial_values"]) - 1):
                ode_definitions_tmp[ode_lhs + "__d" * (order - 1)] = ode_lhs + "__d" * (order - 1)
                state_start_values_tmp[ode_lhs + "__d" * (order - 1)] = ode["initial_values"][order]
            state_start_values_tmp[ode_lhs + "__d" * (max_order - 1)] = ode["initial_values"][(max_order - 1)]
            ode_definitions_tmp[ode_lhs + "__d" * (max_order - 1)] = ode["definition"].replace("'", "__d")

            if "upper_bound" in ode:
                thresholds_tmp.append(ode["symbol"] + " > " + ode["upper_bound"])
            if "lower_bound" in ode:
                thresholds_tmp.append(ode["symbol"] + " < " + ode["upper_bound"])

        # for the interaction with GSL we assume that all state variables
        # (keys in ode_definitions_tmp and initial_values_tmp) correspond to an
        # `y_N` entry, where `N` is counted up from 0 and corresponds to
        # the index into a sorted array of the ODE symbols.

        # compute unique mapping of state variables to corresponding `y__N` variables
        state_variables = sorted(ode_definitions_tmp.keys())
        state_variable_to_y = {s:"y__%i"%i for i,s in enumerate(state_variables)}


        self.thresholds = []
        # Replace all occurrences of state variables in all ode definitions through corresponding `y__N`
        for state_variable_to_map in state_variable_to_y:
            for state_variable in ode_definitions_tmp:
                ode_definition = ode_definitions_tmp[state_variable]
                matcher = re.compile(r"\b(" + state_variable_to_map + r")\b")

                ode_definitions_tmp[state_variable] = matcher.sub(state_variable_to_y[state_variable_to_map], ode_definition)

        for threshold in thresholds_tmp:
            for state_variable_to_map in state_variable_to_y:
                matcher = re.compile(r"\b(" + state_variable_to_map + r")\b")
                threshold = matcher.sub(state_variable_to_y[state_variable_to_map], threshold)
            self.thresholds.append(threshold)

        self.ode_definitions = {}
        self.initial_values = {}
        self.state_start_values = {}

        for state_variable_to_map in state_variable_to_y:
            self.ode_definitions[state_variable_to_y[state_variable_to_map]] = ode_definitions_tmp[state_variable_to_map]

            if state_variable_to_map in initial_values_tmp:
                self.initial_values[state_variable_to_y[state_variable_to_map]] = initial_values_tmp[state_variable_to_map]

            if state_variable_to_map in state_start_values_tmp:
                self.state_start_values[state_variable_to_y[state_variable_to_map]] = state_start_values_tmp[state_variable_to_map]

        self.ode_rhs = [compile(self.ode_definitions[k], "<string>", "eval") for k in sorted(self.ode_definitions.keys())]


    def _prepare_jacobian_matrix(self):
        """Compute the jacobian matrix for the current ODE system.

        First, this function creates a SymPy symbol for each of the state
        variables and each of their definitions. It then creates the
        Jacobian matrix by creating rows that contain the derivations of
        the right-hand sides with respect to every state variable.

        :return: the Jacobian matrix as a list of lists. Every entry
        of the matrix is an expression that can be evaluated in the
        `jacobian` function.

        """

        odes = sorted(self.ode_definitions.items())
        state_vars = [parse_expr(k, local_dict=self.parameters) for k,v in odes]
        ode_defs = [parse_expr(v, local_dict=self.parameters) for k,v in odes]

        self.jacobian_matrix = [
            [compile(str(sympy.diff(rhs, state_var)), "<string>", "eval") for state_var in state_vars]
            for rhs in ode_defs
        ]


    def check_stiffness(self, sim_resolution=0.1, sim_time=20.0, accuracy=1e-3, spike_rate=10.0*1000):
        """
        Performs the test of the ODE system defined in `json_input`. The idea is not
        to compare if the given implicit method or the explicit method is better suited for this small
        simulation but to check for tendencies of stiffness. If we find that the average step size of
        the implicit evolution method is alot larger than the average step size of the explicit method
        this points to the fact that this ODE system could be stiff. Especially that, when a different
        step size is used or when parameters are changed, it become significantly more stiff and an
        implicit evolution scheme could become increasingly important. It is important to note here, that
        this analysis depends significantly on the size of parameters that are assigned for an ODE system
        If these are changed significantly in magnitude the result of the analysis is also changed significantly.

        :param json_input: A list with ode, shape odes and parameters
        """

        step_min_imp, step_average_imp, runtime_imp = self.evaluate_integrator_imp(
            sim_resolution, accuracy, spike_rate, sim_time)

        step_min_exp, step_average_exp, runtime_exp = self.evaluate_integrator_exp(
            sim_resolution, accuracy, spike_rate, sim_time)

        #print("runtime (imp:exp): %f:%f" % (runtime_imp, runtime_exp))

        return self.draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp)


    def _generate_spikes(self, sim_time, sim_resolution, rate):
        """The function computes representative spike trains for the given
        simulation length. Uses a poisson distribution to create
        biologically realistic characteristics of the spike-trains

        :param sim_time: The time of the simulation in ms
        :param sim_resolution: The length of the particular grid
        :return: A list with bins which contain the number of spikes which happened in the corresponding bin.

        """

        sim_time_in_sec = sim_time * 0.001
        sim_resolution_in_sec = sim_resolution * 0.001

        mean_ISI = 1. / rate

        times = []

        t_sum = 0
        while t_sum < sim_time_in_sec:
            t_sum += numpy.random.exponential(mean_ISI, 1)[0]
            times.append(t_sum)

        '''
        Note that besides numpy.random, there is also the independent module
        random. ISIs could also have been drawn using
        random.expovariate(rate).
        '''
        n_spikes = numpy.random.poisson(rate * sim_time_in_sec)
        times = numpy.random.uniform(0, sim_time_in_sec, n_spikes)
        spikes = numpy.sort(times)

        time_slots = int(math.ceil(sim_time_in_sec / sim_resolution_in_sec))
        spikes_per_slot = [0] * time_slots
        for slot in range(0, time_slots):
            t = list(filter(lambda x: slot * sim_resolution_in_sec <= x < (slot + 1) * sim_resolution_in_sec, spikes))
            spikes_per_slot[slot] = len(t)

        return [spikes_per_slot] * len(self.ode_definitions)


    def evaluate_integrator_imp(self, sim_resolution, accuracy, spike_rate, sim_time, raise_errors=True):

        integrator = odeiv.step_bsimp     # Bulirsh-Stoer
        return self.evaluate_integrator(sim_resolution, integrator, accuracy, spike_rate, sim_time, raise_errors)


    def evaluate_integrator_exp(self, sim_resolution, accuracy, spike_rate, sim_time, raise_errors=True):

        integrator = odeiv.step_rk4       # Runge Kutta 4
        return self.evaluate_integrator(sim_resolution, integrator, accuracy, spike_rate, sim_time, raise_errors)


    def evaluate_integrator(self, h, integrator, accuracy, spike_rate, sim_time, raise_errors=True):
        """
        This function computes the average step size and the minimal step size that a given
        integration method from GSL uses to evolve a certain system of ODEs during a certain
        simulation time, integration method from GSL and spike train for a given maximal stepsize.
        :param h: The maximal stepsize for one evolution step in miliseconds
        :param integrator: A method from the GSL library for evolving ODES, e.g. rk4
        :param y: The 'state variables' in f(y)=y'
        :return: Average and minimal step size.
        """

        s_min = h  # the minimal step size cannot be larger than the maximal stepsize h
        simulation_slices = int(round(sim_time / h))

        numpy.random.seed(42)
        spikes = self._generate_spikes(sim_time, h, spike_rate)
        y = self._compute_initial_state_vector()

        self._prepare_jacobian_matrix()

        gsl_stepper = integrator(len(y), self.step, self.jacobian)
        control = odeiv.control_y_new(gsl_stepper, accuracy, accuracy)
        evolve = odeiv.evolve(gsl_stepper, control, len(y))

        t = 0.0
        step_counter = 0
        sum_last_steps = 0
        s_min_old = 0
        runtime = 0.0
        s_min_lower_bound = 5e-09

        for time_slice in range(simulation_slices):
            t_new = t + h
            counter_while_loop = 0
            t_old = 0
            while t < t_new:
                counter_while_loop += 1
                t_old = t
                time_start = time.time()
                try:
                    # h_ is NOT the reached step size but the suggested next step size!
                    t, h_, y = evolve.apply(t, t_new, h, y)
                except Exception as e:
                    print("     ===> Failure of %s at t=%.2f with h=%.2f (y=%s)" % (gsl_stepper.name(), t, h, y))
                    if raise_errors:
                        raise
                runtime += time.time() - time_start
                step_counter += 1
                s_min_old = s_min
                s_min = min(s_min, t - t_old)
                if s_min < s_min_lower_bound:
                    estr = "Integration step below %.e (s=%.f). Please check your ODE." % (s_min_lower_bound, s_min)
                    if raise_errors:
                        raise Exception(estr)
                    else:
                        print(estr)

            if counter_while_loop > 1:
                step_counter -= 1
                sum_last_steps += t_new - t_old
                # it is possible that the last step in a simulation_slot is very small, as it is simply
                # the length of the remaining slot. Therefore we don't take the last step into account
                s_min = s_min_old

            threshold_crossed = False
            for threshold in self.thresholds:
                _globals = self.math_module_funcs.copy()
                local_parameters = self.parameters.copy()
                local_parameters.update({"y__%i"%i: y for i,y in enumerate(y)})
                if eval(threshold, _globals, local_parameters):
                    threshold_crossed = True
                    break  # break inner loop

            if threshold_crossed:  # break outer loop
                break

            for idx, initial_value in enumerate(self.initial_values):
                matcher = re.compile(r".*(\d)+$")
                oder_order_number = int(matcher.match(initial_value).groups(0)[0])

                # TODO: Why is there no convolution here? Is it correct
                # and meaningful to just sum up the number of spikes?
                _globals = self.math_module_funcs.copy()
                y[oder_order_number] += eval(self.initial_values[initial_value], _globals, self.parameters) * spikes[idx][time_slice]

        step_average = (t - sum_last_steps) / step_counter
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

        machine_precision = numpy.finfo(float).eps

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


    def _compute_initial_state_vector(self):
        """
        Computes an numpy vector start values for the ode system.
        :return: Numpy-Vector with `dimension` elements
        """
        _globals = self.math_module_funcs.copy()

        y = numpy.zeros(len(self.ode_definitions), numpy.float)
        for state_start_variable in self.state_start_values:
            matcher = re.compile(r".*(\d)+$")
            oder_order_number = int(matcher.match(state_start_variable).groups(0)[0])

            local_parameters = locals().copy()
            local_parameters.update(self.parameters)

            y[oder_order_number] = eval(self.state_start_values[state_start_variable], _globals, local_parameters)
        return y


    def jacobian(self, t, y, params):
        """Callback function that compute the jacobian matrix for the current
        state vector `y`.

        :param t: current time in the step (from 0 to step_size)
        :param y: the current state vector of the ODE system
        :param params: Prescribed GSL parameters (not used here).

        :return: dfdy that contains the jacobian matrix with respect
        to y`dfdt` is not computed and set to zero matrix now.

        """
        dimension = len(y)
        dfdy = numpy.zeros((dimension, dimension), numpy.float)
        dfdt = numpy.zeros((dimension,))

        _globals = self.math_module_funcs.copy()

        # evaluate every entry of the `jacobian_matrix` and store the
        # result in the corresponding entry of the `dfdy`
        for row in range(0, dimension):
            for col in range(0, dimension):
                local_parameters = self.parameters.copy()
                local_parameters.update({"y__%i"%i: y for i,y in enumerate(y)})
                dfdy[row, col] = eval(self.jacobian_matrix[row][col], _globals, local_parameters)

        return dfdy, dfdt


    def step(self, t, y, params):
        """Callback function to compute an integration step.

        Please note that this function is somewhat dangerous as it uses
        `eval` to execute the definition of the ODEs and entails the risk
        of code injection.

        :param t: current time in the step (from 0 to step_size)
        :param y: the current state vector of the ODE system
        :param params: Prescribed GSL parameters (not used here).

        :return: Updated state vector

        """
        local_parameters = self.parameters.copy()
        local_parameters.update({"y__%i"%i: y for i,y in enumerate(y)})

        _globals = self.math_module_funcs.copy()

        try:
            return [eval(ode, _globals, local_parameters) for ode in self.ode_rhs]
        except Exception as e:
            print("E==>", type(e).__name__ + ": " + str(e))
            print("     Local parameters at time of failure:")
            for k,v in local_parameters.items():
                print("    ", k, "=", v)
            raise
