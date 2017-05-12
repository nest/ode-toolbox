import Numeric
import jinja2
from math import *
import pygsl.odeiv as odeiv
from sympy.parsing.sympy_parser import parse_expr # this module is used in the generated code
from sympy import *
import re
import numpy as np  # this module is used in the generated code
import numpy.random

# for the testing purpose fix the seed to 42 in order to make results reproducable
numpy.random.seed(42)

# the following variables must be defined globally since they are accessed from the step, jacobian, threshold and
# these functions are called from the framework and cannot take these variables as parameters
state_variables = []
ode_definitions = []
function_variables = []
function_variable_definitions = []
step_function_implementation = None
jacobian_function_implementation = None
dimension = 0


def check_ode_system_for_stiffness(odes_and_function_variables, default_values, threshold_body):
    """
    Performs the test of the ODE system defined in `odes_and_function_variables`
    :param odes_and_function_variables: A list with odes and functions 
    :param default_values: a list with variable declrations that 
    :param threshold_body: defines a boolean statement that becomes true if the
    """
    # Process input
    global state_variables
    global ode_definitions
    global function_variables
    global function_variable_definitions

    function_variables = []
    function_variable_definitions = []
    state_variables = []
    ode_definitions = []

    parse_input_parameters(function_variable_definitions,
                           function_variables,
                           ode_definitions,
                           odes_and_function_variables,
                           state_variables)
    # `default_values` contains a series of variables from the differential equations declaration separated by the
    # newline they are defined in the global scope so that different functions can access them either
    for default_value in default_values:
        exec (default_value) in globals()

    global dimension
    dimension = len(ode_definitions)

    prepare_jacobian_matrix(function_variable_definitions, function_variables, ode_definitions, state_variables)
    prepare_step_function(function_variable_definitions, function_variables, ode_definitions, state_variables)

    # define simulation time in seconds and milliseconds
    sim_time = 20.  # in ms
    sim_time_in_sec = sim_time * 0.001

    slot_width = 0.2  # slot width in ms
    slot_width_in_sec = slot_width * 0.001

    gen_inh = generate_representative_spike_train(sim_time_in_sec, slot_width_in_sec)

    simulation_slots = int(round(sim_time / slot_width))

    print("#### SUMMARY ####")
    print("State variables:\n" + str(state_variables))
    print("ODE definitions:")
    for ode in ode_definitions:
        print(ode)
    print("#### END ####")

    print("Starts stiffness test for the ODE system...")
    print ("######### imp #########")
    step_min_imp, step_average_imp = evaluate_integrator(
        slot_width,
        sim_time,
        simulation_slots,
        odeiv.step_bsimp,
        step,
        jacobian,
        [gen_inh] * dimension,
        start_values,
        initial_values,
        threshold_body)

    # `default_values` contains a series of variables from the differential equations declaration separated by the
    # newline they are defined in the global scope so that different functions can access them either
    for default_value in default_values:
        exec (default_value) in globals()
    print ("######### rk expl ########")
    step_min_exp, step_average_exp = evaluate_integrator(
        slot_width,
        sim_time,
        simulation_slots,
        odeiv.step_rk4,
        step,
        jacobian,
        [gen_inh] * dimension,
        start_values,
        initial_values,
        threshold_body)

    print ("######### results #######")
    print "min_imp: {} min_exp: {}".format(step_min_imp, step_min_exp)
    print "avg_imp: {} avg_exp: {}".format(step_average_imp, step_average_exp)
    print ("########## end ##########")


def prepare_step_function(function_variable_definitions, function_variables, ode_definitions, state_variables):
    for idx in range(0, len(ode_definitions)):
        ode_definitions[idx] = replace_state_variables_through_array_access(ode_definitions[idx])
    for idx in range(0, len(function_variable_definitions)):
        function_variable_definitions[idx] = replace_state_variables_through_array_access(
            function_variable_definitions[idx])

    # This computation is done here only due to performance considerations
    global step_function_implementation
    step_function_body = (
        "{% for var, defining_expr in function_variables %}\n"
        "{{var}} = {{defining_expr}}\n"
        "{% endfor %}\n"
        "{% for ode_var, ode in odes %}\n"
        "{{ode_var}} = {{ode}}\n"
        "{% endfor %}\n")
    step_function_implementation = jinja2.Template(step_function_body)
    step_function_implementation = step_function_implementation.render(
        odes=zip(state_variables_to_f(state_variables), ode_definitions),
        function_variables=zip(function_variables, function_variable_definitions))
    print "step function implementation:"
    print step_function_implementation
    # compile the code to boost the performance
    step_function_implementation = compile(step_function_implementation, '<string>', 'exec')


def prepare_jacobian_matrix(function_variable_definitions, function_variables, ode_definitions, state_variables):
    global jacobian_function_implementation
    jacobian_function_body = (
        "{% for var, defining_expr in function_variables %}\n"
        "{{var}} = parse_expr('{{defining_expr}}', local_dict=locals())\n"
        "{% endfor %}\n"
        "state_variable_expr = []\n"
        "right_hand_sides_expr = []\n"
        "{% for state_var, defining_expr in odes %}\n"
        "state_variable_expr.append(parse_expr('{{state_var}}', local_dict=locals()))\n"
        "right_hand_sides_expr.append(parse_expr('{{defining_expr}}', local_dict=locals()))\n"
        "{% endfor %}\n"
        "for rhs in right_hand_sides_expr:\n"
        "    row = []\n"
        "    result_matrix_str = ''\n"
        "    for state_variable in state_variable_expr:\n"
        "        result_matrix_str += str(diff(rhs, state_variable)) + ' '\n"
        "        row.append(replace_state_variables_through_array_access(str(diff(rhs, state_variable))))\n"
        "    print(result_matrix_str)\n"
        "    result_matrix.append(row)\n"

    )
    jacobian_function_implementation = jinja2.Template(jacobian_function_body)
    jacobian_function_implementation = jacobian_function_implementation.render(
        odes=zip(state_variables, ode_definitions),
        function_variables=zip(function_variables, function_variable_definitions))
    print "jacobian function implementation:"
    print jacobian_function_implementation
    jacobian_function_implementation = compile(jacobian_function_implementation, '<string>', 'exec')
    # this matrix is used in the jacobian function
    global jacobian_matrix
    jacobian_matrix = calculate_jacobian()


def parse_input_parameters(function_variable_definitions,
                           function_variables,
                           ode_definitions,
                           odes_and_function_variables,
                           state_variables):
    for entry in odes_and_function_variables:
        if entry.startswith("function"):
            # e.g. 'function a = b'
            # extract assignment
            assig = entry[len("function"):]
            variable = assig.split("=")[0].strip()
            defining_expression = assig.split("=")[1].strip()
            function_variables.append(variable)
            function_variable_definitions.append(defining_expression)

        else:
            lhs = entry.split("=")[0].strip()
            lhs = lhs.replace('f', 'y')
            rhs = entry.split("=")[1].strip()

            state_variables.append(lhs)
            ode_definitions.append(rhs)


def replace_state_variables_through_array_access(definition):
    match_f_or_y = re.compile(r"\b(y|f)_(\d)+\b")
    result = match_f_or_y.sub(r"\1[\2]", definition)
    return result


def state_variables_to_f(state_variable):
    result = []
    for idx in range(0, len(state_variable)):
        result.append("f[" + str(idx) + "]")
    return result


def step(t, y, params):
    f = Numeric.zeros((dimension,), Numeric.Float)
    exec (step_function_implementation)
    return f


def threshold(y, threshold_body):
    threshold_body = replace_state_variables_through_array_access(threshold_body)
    return eval(threshold_body)


def generate_representative_spike_train(sim_time_in_sec, slot_width_in_sec):
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
    # return spikes_per_slot
    spike_train = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    return spike_train * time_slots


def evaluate_integrator(h,
                        sim_time,
                        simulation_slots,
                        integrator,
                        step_function,
                        jacobian,
                        spikes,
                        y,
                        initial_values,
                        threshold_body):
    s_min = h  # the minimal step size cannot be larger than the maximal stepsize h

    gls_stepper = integrator(len(y), step_function, jacobian)  # must be jaccobian
    control = odeiv.control_y_new(gls_stepper, 1e-2, 1e-2)
    evolve = odeiv.evolve(gls_stepper, control, len(y))

    t = 0.0
    step_counter = 0
    for time_slot in range(simulation_slots):
        t_new = t + h
        print "Start while loop at slot " + str(time_slot)
        while t < t_new:
            t_old = t
            t, h_, y = evolve.apply(t, t_new, h, y)  # h_ is NOT the reached step size but the suggested next step size!
            step_counter += 1
            s_min_old = s_min
            s_min = min(s_min, t - t_old)
            print str(time_slot) + ":   t=%.15f, current stepsize=%.15f y=" % (t, t - t_old), y
            if s_min < 0.000005:
                raise Exception("Check your ODE system. The integrator step becomes to small "
                                "in order to support reasonable simulation")
        s_min = s_min_old

        print "End while loop"

        if threshold is not None and threshold(y, threshold_body):
            print("The predefined threshold is crossed. Terminate the evaluation procedure.")
            break

        for idx, initial_value in enumerate(initial_values):
            y[idx] += initial_value * spikes[idx][time_slot]
    step_average = sim_time / step_counter
    return s_min_old, step_average


def calculate_jacobian():
    result_matrix = []
    exec(jacobian_function_implementation)
    return result_matrix


def jacobian(t, y, t_m):
    dfdy = Numeric.zeros((dimension, dimension), Numeric.Float)
    dfdt = Numeric.zeros((dimension,))

    # since the expression is put into a lambda function being evaluated it cannot access function's parameter
    # therefore, add the `y` variable to the globals manually
    g = globals()
    g["y"] = y

    for row in range(0, dimension):
        for col in range(0, dimension):
            tmp = eval("lambda: " + jacobian_matrix[row][col], g)
            dfdy[row, col] = tmp()

    return dfdy, dfdt



