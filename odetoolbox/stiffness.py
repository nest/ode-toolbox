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

import numpy
import jinja2
from math import *
from sympy import *
import re
import numpy.random
import time

# The change of stderr is a hack to silence PyGSL's error printing,
# which often gives false hints on what is actually going on.
import sys
import os
oldstderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    import pygsl.odeiv as odeiv
except ImportError as ie:
    sys.stderr.close()
    sys.stderr = oldstderr
    print("Warning: PyGSL is not available. The stiffness test will be skipped.")
    print("Warning: " + str(ie), end="\n\n\n")
    raise

sys.stderr.close()
sys.stderr = oldstderr

# for the testing purpose fix the seed to 42 in order to make results reproducible
numpy.random.seed(42)

# the following variables must be defined globally since they are accessed from the step, jacobian, threshold and
# these functions are called from the framework and cannot take these variables as parameters
step_function_implementation = None
jacobian_matrix_implementation = None
parameters = globals().copy()


def check_ode_system_for_stiffness(json_input):
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

    global step_function_implementation
    global jacobian_matrix_implementation
    global parameters
    step_function_implementation = None
    jacobian_matrix_implementation = None

    ode_definitions, initial_values, state_start_values, thresholds = prepare_for_evaluate_integrator(json_input)

    # `parameters` contains a series of variables from the differential equations declaration separated by the
    # newline they are defined in the global scope so that different functions can access them either
    for variable in json_input["parameters"]:
        parameters[variable] = float(json_input["parameters"][variable])

    jacobian_matrix_implementation = prepare_jacobian_matrix(ode_definitions)
    step_function_implementation = prepare_step_function(ode_definitions)

    # define simulation time in seconds and milliseconds
    sim_time = 20.  # in ms
    sim_time_in_sec = sim_time * 0.001

    # define the length of the slotwidth in seconds and milliseconds
    slot_width = 0.2  # slot width in ms
    slot_width_in_sec = slot_width * 0.001

    gen_inh = generate_spikes(sim_time_in_sec, slot_width_in_sec)
    
    # calculate the amount of simulation slots
    simulation_slots = int(round(sim_time / slot_width))
    dimension = len(ode_definitions)

    # our aim is not to compare to evolution methods but to find tendancies of stiffness for the system
    # therefore we use one implicit evolution method, here the bulirsh stoer method and one explicit method
    imp_solver = odeiv.step_bsimp
    # print ("######### {} #########".format(imp_solver.__name__))
    step_min_imp, step_average_imp, runtime_imp = evaluate_integrator(
        h=slot_width,
        simulation_slices=simulation_slots,
        integrator=imp_solver,
        step_function=step,
        jacobian=jacobian,
        spikes=[gen_inh] * dimension,
        y=compute_initial_state_vector(state_start_values, dimension),
        initial_values=initial_values,
        thresholds=thresholds)

    # the explicit evolution method is a runge kutta 4
    exp_solver = odeiv.step_rk4
    # print ("######### {} #########".format(exp_solver.__name__))
    step_min_exp, step_average_exp, runtime_exp = evaluate_integrator(
        h=slot_width,
        simulation_slices=simulation_slots,
        integrator=exp_solver,
        step_function=step,
        jacobian=jacobian,
        spikes=[gen_inh] * dimension,
        y=compute_initial_state_vector(state_start_values, dimension),
        initial_values=initial_values,
        thresholds=thresholds)

    # print ("######### results #######")
    # print "min_{}: {} min_{}: {}".format(imp_solver.__name__, step_min_imp, exp_solver.__name__, step_min_exp)
    # print "avg_{}: {} avg_{}: {}".format(imp_solver.__name__, step_average_imp, exp_solver.__name__, step_average_exp)
    # print ("########## end ##########")

    print("runtime (imp:exp): %f:%f" % (runtime_imp, runtime_exp))
    return draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp)


def prepare_for_evaluate_integrator(json_input):
    """
    :param json_input: A JSON object that encodes ode system.
    :return: `ode_definitions`, `initial_values`, `state_start_values` which store a representation of the ODE system
             that can be propagated by the GSL. `thresholds` which stores a list with thresholds that can be evaluated
             in the `evaluate_integrator` method
    """
    ode_definitions = {}
    initial_values = {}
    state_start_values = {}
    thresholds = []

    # odes are stored in shapes and
    for shape in json_input["shapes"]:
        shape_name = shape["symbol"]
        max_order = len(shape["initial_values"])
        for order in range(0, len(shape["initial_values"]) - 1):
            ode_definitions[shape_name + "__d" * (order - 1)] = shape_name + "__d" * (order - 1)
            initial_values[shape_name + "__d" * order] = shape["initial_values"][order]
        initial_values[shape_name + "__d" * (max_order - 1)] = shape["initial_values"][(max_order - 1)]
        ode_definitions[shape_name + "__d" * (max_order - 1)] = shape["definition"].replace("'", "__d")

    # we omit initial values for odes since they cannot be connected to buffers
    for ode in json_input["odes"]:
        ode_lhs = ode["symbol"]
        max_order = len(ode["initial_values"])
        for order in range(0, len(ode["initial_values"]) - 1):
            ode_definitions[ode_lhs + "__d" * (order - 1)] = ode_lhs + "__d" * (order - 1)
            state_start_values[ode_lhs + "__d" * (order - 1)] = ode["initial_values"][order]
        state_start_values[ode_lhs + "__d" * (max_order - 1)] = ode["initial_values"][(max_order - 1)]
        ode_definitions[ode_lhs + "__d" * (max_order - 1)] = ode["definition"].replace("'", "__d")

        if "upper_bound" in ode:
            thresholds.append(ode["symbol"] + " > " + ode["upper_bound"])
        if "lower_bound" in ode:
            thresholds.append(ode["symbol"] + " < " + ode["upper_bound"])

    # for the interaction with GSL we assume that all state variables(keys in ode_definitions, initial_values)
    # correspond to an `y_N` entry, where `N` is an appropriate number.

    # compute unique mapping of state variables to corresponding `y__N` variables
    state_variables = sorted(ode_definitions.keys())

    state_variable_to_y = {}
    for i, state_variable in enumerate(state_variables):
        state_variable_to_y[state_variable] = "y__" + str(i)

    thresholds_tmp = []
    # Replace all occurrences of state variables in all ode definitions through corresponding `y__N`
    for state_variable_to_map in state_variable_to_y:
        for state_variable in ode_definitions:
            ode_definition = ode_definitions[state_variable]
            matcher = re.compile(r"\b(" + state_variable_to_map + r")\b")

            ode_definitions[state_variable] = matcher.sub(state_variable_to_y[state_variable_to_map], ode_definition)

    for threshold in thresholds:
        for state_variable_to_map in state_variable_to_y:
            matcher = re.compile(r"\b(" + state_variable_to_map + r")\b")
            threshold = matcher.sub(state_variable_to_y[state_variable_to_map], threshold)
            threshold = replace_state_variables_through_array_access(threshold)
        thresholds_tmp.append(threshold)

    ode_definitions_tmp = {}
    initial_values_tmp = {}
    state_start_values_tmp = {}

    for state_variable_to_map in state_variable_to_y:
        ode_definitions_tmp[state_variable_to_y[state_variable_to_map]] = ode_definitions[state_variable_to_map]

        if state_variable_to_map in initial_values:
            initial_values_tmp[state_variable_to_y[state_variable_to_map]] = initial_values[state_variable_to_map]

        if state_variable_to_map in state_start_values:
            state_start_values_tmp[state_variable_to_y[state_variable_to_map]] = state_start_values[state_variable_to_map]

    return ode_definitions_tmp, initial_values_tmp, state_start_values_tmp, thresholds_tmp


def prepare_jacobian_matrix(ode_definitions):
    """
    Compute the jacobian matrix for the current ODE system.
    :param: ode_definitions Map of ode LHSs and definitions
    :return: jacobian_matrix Stores the entries of the jacobian matrix as list of lists. Every entry of the matrix is
                             an expression that can evaluated in the `jacobian` function.
    """

    # defines the implementation of the jacobian matrix computation as an jinja2 template
    # Logic:
    # 1) for all function variables create a SymPy variable that stores its definition as a SymPy-expression
    # 2) For every ODE: create a SymPy symbol for every lefthandside of the ode and create a SymPy-expression
    # 3) Create the jacobian matrix as:
    #     every row is composed of the derivation of the righthandside of one with respect to every state variable from
    #     `state_variables`
    #     repeat this procedure for every ODE from `ode_definitions`
    jacobian_function_body = (
        "from sympy.parsing.sympy_parser import parse_expr\n"
        "state_variable_expr = []\n"
        "right_hand_sides_expr = []\n"
        "{% for state_var in state_variables %}\n"
        "state_variable_expr.append(parse_expr('{{state_var}}', local_dict=parameters))\n"
        "right_hand_sides_expr.append(parse_expr('{{odes[state_var]}}', local_dict=parameters))\n"
        "{% endfor %}\n"
        "result_matrix_str = ''\n"
        "for rhs in right_hand_sides_expr:\n"
        "    row = []\n"
        "    for state_variable in state_variable_expr:\n"
        "        result_matrix_str += str(diff(rhs, state_variable)) + ' '\n"
        "        row.append(replace_state_variables_through_array_access(str(diff(rhs, state_variable))))\n"
        "    result_matrix_str = result_matrix_str + '\\n'\n"
        "    result_matrix.append(row)\n")

    # map state variables to a create jinja2 template
    jacobian_function_implementation = jinja2.Template(jacobian_function_body)
    jacobian_function_implementation = jacobian_function_implementation.render(
        odes=ode_definitions,
        state_variables=sorted(ode_definitions.keys()))

    # compile the generated code from the  jinja2 template for the better performance
    jacobian_function_implementation = compile(jacobian_function_implementation, '<string>', 'exec')
    # this matrix which is stored in the global variable is used in the `jacobian` function
    jacobian_matrix = calculate_jacobian(jacobian_function_implementation)

    return jacobian_matrix


def calculate_jacobian(jacobian_function_implementation):
    """
    Computes the specific jacobian matrix for current system of odes
    This function uses the following global variables: `state_variables`, `ode_definitions`, `function_variables`,
    `function_variable_definitions`
    :return: result_matrix that stores the entries of the jacobian matrix as list of lists. Every entry of the matrix is
             an expression that can evaluated in the `jacobian` function.
    """
    # both variables are set during the execution of the `jacobian_function_implementation`
    result_matrix = []
    result_matrix_str = ""
    exec(jacobian_function_implementation)
    print("\nCalculated jacobian matrix: ")
    # `result_matrix_str` is calculated as a part of the `jacobian_function_implementation`
    print("\n" + result_matrix_str)
    return result_matrix


def prepare_step_function(ode_definitions):
    """
    Create the implementation of the  step function used in the GSL framework to compute one integration step.
    :param: ode_definitions Map of ode LHSs and definitions

    :return: step_function_implementation Stores a compiled Python-code that performs one integration step. 
    """
    # The input provided as the function parameters is not valid Python/GSL code. Convert all references of f_n and y_n
    # to f[n] and y[n]
    ode_definitions_tmp = {}
    for state_variable in ode_definitions:
        ode_definitions_tmp[replace_state_variables_through_array_access(state_variable.replace("y", "f"))] = \
            replace_state_variables_through_array_access(ode_definitions[state_variable])

    # defines the implementation of the jacobian matrix computation as an jinja2 template

    step_function_body = (
        "{% for ode_var in ode_vars %}\n"
        "{{ode_var}} = {{odes[ode_var]}}\n"
        "{% endfor %}\n")

    step_function_implementation = jinja2.Template(step_function_body)
    step_function_implementation = step_function_implementation.render(
        odes=ode_definitions_tmp,
        ode_vars=sorted(ode_definitions_tmp.keys()))
    print("step function implementation:")
    print(step_function_implementation)

    # compile the code to boost the performance
    step_function_implementation = compile(step_function_implementation, '<string>', 'exec')
    return step_function_implementation


def replace_state_variables_through_array_access(definition):
    """
    Convert all references of y_n in `definition` to y[n]
    :param definition: String to convert
    :return: Converted string
    """
    match_f_or_y = re.compile(r"\b([yf])__(\d)+\b")
    result = match_f_or_y.sub(r"\1[\2]", definition)
    return result


def state_variables_to_f(state_variable):
    """
    Converts all state variables to a list composed of `f[n]`, n in {0 to len(state_variables)}. This list is used
    to generate the lefthandside of the assignments of ODEs in the `step`-function
    :param state_variable: 
    :return: 
    """
    result = []
    for idx in range(0, len(state_variable)):
        result.append("f[" + str(idx) + "]")
    return result


def generate_spikes(sim_time_in_sec, slot_width_in_sec):
    """
    The function computes representative spike trains for the given simulation length. Uses a poisson distribution to 
    create biologically realistic characteristics of the spike-trains
    :param sim_time_in_sec: The time of the simulation in seconds 
    :param slot_width_in_sec: The length of the particular grind
    :return: A list with bins which contain the number of spikes which happened in the corresponding bin.
    """
    rate = 10. * 1000.
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

    time_slots = int(ceil(sim_time_in_sec / slot_width_in_sec))
    spikes_per_slot = [0] * time_slots
    for slot in range(0, time_slots):
        t = list(filter(lambda x: slot * slot_width_in_sec <= x < (slot + 1) * slot_width_in_sec, spikes))
        spikes_per_slot[slot] = len(t)
    return spikes_per_slot


def evaluate_integrator(h,
                        simulation_slices,
                        integrator,
                        step_function,
                        jacobian,
                        spikes,
                        y,
                        initial_values,
                        thresholds):
    """
    This function computes the average step size and the minimal step size that a given 
    integration method from GSL uses to evolve a certain system of ODEs during a certain
    simulation time, integration method from GSL and spike train for a given maximal stepsize.
    :param h: The maximal stepsize for one evolution step in miliseconds
    :param integrator: A method from the GSL library for evolving ODES, e.g. rk4
    :param step_function: The function f that defines the system of ODEs as f(y)= y' that is evolved by the GSL method
    :param jacobian: the jacobian matrix of the function `step_function`
    :param spikes: A representative spike train for the given simulation time `sim_time`
    :param y: The 'state variables' in f(y)=y'
    :return: Average and minimal step size.
    """
    s_min = h  # the minimal step size cannot be larger than the maximal stepsize h

    gls_stepper = integrator(len(y), step_function, jacobian)  # must be jacobian
    control = odeiv.control_y_new(gls_stepper, 1e-3, 1e-3)
    evolve = odeiv.evolve(gls_stepper, control, len(y))

    t = 0.0
    step_counter = 0
    sum_last_steps = 0
    s_min_old = 0
    runtime = 0.0

    for time_slice in range(simulation_slices):
        t_new = t + h
        # print "Start while loop at slot " + str(time_slice)
        counter_while_loop = 0
        t_old = 0
        while t < t_new:
            counter_while_loop += 1
            t_old = t
            time_start = time.time()
            t, h_, y = evolve.apply(t, t_new, h, y)  # h_ is NOT the reached step size but the suggested next step size!
            runtime += time.time() - time_start
            step_counter += 1
            s_min_old = s_min
            s_min = min(s_min, t - t_old)
            # print str(time_slice) + ":   t=%.15f, current stepsize=%.15f y=" % (t, t - t_old), y
            if s_min < 0.000000005:
                raise Exception("Check your ODE system. The integrator step becomes to small in order to support "
                                "reasonable simulation", s_min)
        if counter_while_loop > 1:
            step_counter -= 1
            sum_last_steps += t_new - t_old
            # it is possible that the last step in a simulation_slot is very small, as it is simply
            # the length of the remaining slot. Therefore we don't take the last step into account        
            s_min = s_min_old

        # print "End while loop"
        threshold_crossed = False
        for threshold in thresholds:
            parameters_with_locals = parameters.copy()
            parameters_with_locals["y"] = y
            if eval(threshold, parameters_with_locals):
                threshold_crossed = True
                break  # break inner loop

        if threshold_crossed:  # break outer loop
            break

        for idx, initial_value in enumerate(initial_values):
            matcher = re.compile(r".*(\d)+$")
            oder_order_number = int(matcher.match(initial_value).groups(0)[0])
            parameters_with_locals = globals().copy()
            for key in parameters:
                parameters_with_locals[key] = parameters[key]
            y[oder_order_number] += eval(initial_values[initial_value], parameters_with_locals) * spikes[idx][time_slice]

    step_average = (t - sum_last_steps) / step_counter
    return s_min_old, step_average, runtime


def draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp):
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
        if step_average_imp > 2*step_average_exp:
            return "implicit"
        else:
            return "explicit"
    else:
        raise Exception("This case cannot happen.")


def step(t, y, params):
    """
    The GSL callback function that computes of integration step.
    :param t: current time in the step (from 0 to step_size)
    :param y: the current state vector of the ODE system
    :param step_function_implementation: global variable that contains the implementation of the step function.
    :param params: Prescribed GSL parameters which are  not used in this step function.
    :return: Updated state vector stored in `f`
    """
    global step_function_implementation
    global parameters
    dimension = len(y)
    # f is used in the step_function_implementation. since it is a return value of this function, it is declared here
    # explicitly
    f = numpy.zeros((dimension,), numpy.float)

    parameters_with_locals = parameters.copy()
    parameters_with_locals["f"] = f
    parameters_with_locals["y"] = y

    exec(step_function_implementation, parameters_with_locals)
    return f


def compute_initial_state_vector(state_start_values, dimension):
    """
    Computes an numpy vector start values for the ode system.
    :param state_start_values:
    :param dimension: The size of the ode system
    :return: Numpy-Vector with `dimension` elements
    """
    y = numpy.zeros(dimension, numpy.float)
    for state_start_variable in state_start_values:
        matcher = re.compile(r".*(\d)+$")
        oder_order_number = int(matcher.match(state_start_variable).groups(0)[0])

        parameters_with_locals = locals().copy()
        parameters_with_locals.update(parameters)

        y[oder_order_number] = eval(state_start_values[state_start_variable], parameters_with_locals)
    return y


def jacobian(t, y, params):
    """
    Callback function that computes the jacobian matrix for the current state vector `y`.
    :param t: another prescribed parameters which are not used here
    :param y: represents the current state vector of the ODE system as updated through the GSL integrator.
    :param params: another prescribed parameters which are not used here
    :return: dfdy that contains the jacobian matrix with respect to y`dfdt` is not computed and set to zero matrix now.
    """
    dimension = len(y)
    dfdy = numpy.zeros((dimension, dimension), numpy.float)
    dfdt = numpy.zeros((dimension,))

    # since the expression is put into a lambda function being evaluated it cannot access function's parameter
    # therefore, add the `y` variable to the globals manually

    # evaluate every entry of the `jacobian_matrix` and store the result in the corresponding entry of the `dfdy`
    for row in range(0, dimension):
        for col in range(0, dimension):
            # wrap the expression to a lambda function for the improved performance
            parameters_with_locals = parameters.copy()
            parameters_with_locals["y"] = y
            tmp = eval("lambda: " + jacobian_matrix_implementation[row][col], parameters_with_locals)
            dfdy[row, col] = tmp()

    return dfdy, dfdt
