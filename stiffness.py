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
dimension = 0


def check_ode_system_for_stiffness(odes_and_function_variables, default_values, threshold_body):
    """
    Performs the test of the ODE system defined in `odes_and_function_variables`. The idea is not
    to compare if the given implicit method or the explicit method is better suited for this small
    simulation but to check for tendencies of stiffness. If we find that the average step size of
    the implicit evolution method is alot larger than the average step size of the explicit method
    this points to the fact that this ODE system could be stiff. Especially that, when a different 
    step size is used or when parameters are changed, it become significantly more stiff and an 
    implicit evolution scheme could become increasingly important. It is important to note here, that
    this analysis depends significantly on the size of parameters that are assigned for an ODE system
    If these are changed significantly in magnitude the result of the analysis is also changed significantly.
    
    :param odes_and_function_variables: A list with ode and function definitions
    :param default_values: a list with variable declarations together with their start values. It must contain to special
                          lists: 'initial_values' and 'start_values' with floating point values. Length of these lists 
                          equals the number of ODEs
     
    :param threshold_body: defines a boolean expression that becomes true if when of membrane potential crosses a 
                           threshold.
    """
    global state_variables
    global ode_definitions
    global function_variables
    global function_variable_definitions

    # reset global variables that can have values from a previous function run
    function_variables = []
    function_variable_definitions = []
    state_variables = []
    ode_definitions = []

    parse_input_parameters(odes_and_function_variables)
    # `default_values` contains a series of variables from the differential equations declaration separated by the
    # newline they are defined in the global scope so that different functions can access them either
    for default_value in default_values:
        exec (default_value) in globals()

    global dimension
    dimension = len(ode_definitions)

    print("#### SUMMARY ####")
    prepare_jacobian_matrix()
    prepare_step_function()

    # define simulation time in seconds and milliseconds
    sim_time = 20.  # in ms
    sim_time_in_sec = sim_time * 0.001

    # define the length of the slotwidth in seconds and milliseconds
    slot_width = 0.2  # slot width in ms
    slot_width_in_sec = slot_width * 0.001

    gen_inh = generate_representative_spike_train(sim_time_in_sec, slot_width_in_sec)
    
    #calculate the amount of simulation slots
    simulation_slots = int(round(sim_time / slot_width))
    print("#### END ####")

    print("Starts stiffness test for the ODE system...")
    # our aim is not to compare to evolution methods but to find tendancies of stiffness for the system
    # therefore we use one implicit evolution method, here the bulirsh stoer method and one explicit method
    imp_solver = odeiv.step_bsimp
    print ("######### {} #########".format(imp_solver.__name__))
    step_min_imp, step_average_imp = evaluate_integrator(
        slot_width,
        sim_time,
        simulation_slots,
        imp_solver,
        step,
        jacobian,
        [gen_inh] * dimension,
        start_values,    # this variable was be added in `exec (default_value) in globals()`
        initial_values,  # this variable was be added in `exec (default_value) in globals()`
        threshold_body)

    # `default_values` contains a series of variables from the differential equations declaration separated by the
    # newline they are defined in the global scope so that different functions can access them either
    for default_value in default_values:
        exec (default_value) in globals()
    # the explicit evolution method is a runge kutta 4
    exp_solver = odeiv.step_rk4
    print ("######### {} #########".format(exp_solver.__name__))
    step_min_exp, step_average_exp = evaluate_integrator(
        slot_width,
        sim_time,
        simulation_slots,
        exp_solver,
        step,
        jacobian,
        [gen_inh] * dimension,
        start_values,    # this variable was be added in `exec (default_value) in globals()`
        initial_values,  # this variable was be added in `exec (default_value) in globals()`
        threshold_body)

    print ("######### results #######")
    print "min_{}: {} min_{}: {}".format(imp_solver.__name__, step_min_imp, exp_solver.__name__, step_min_exp)
    print "avg_{}: {} avg_{}: {}".format(imp_solver.__name__, step_average_imp, exp_solver.__name__, step_average_exp)
    print ("########## end ##########")
    return draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp)


def parse_input_parameters(odes_and_function_variables):
    """    
    :param odes_and_function_variables: A list with ode and function definitions 
    :return: Initializes global variables:`state_variables`, `ode_definitions`, `function_variables`, 
            `function_variable_definitions` 
    """
    global state_variables
    global ode_definitions
    global function_variables
    global function_variable_definitions
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


def prepare_jacobian_matrix():
    """
    Compute the jacobian matrix for the current ODE system.
    :param: global state_variables List of state variables which are passed as global variables
    :param: global ode_definitions List of  ode definitions for every state variable from `state_variables` which are 
                                   passed as global variables
    :param: global function_variables List of state variables which are passed as global variables
    :param: global function_variable_definitions List of  ode definitions for every state variable from 
                                                 `function_variables` which are passed as global variables

    :return: jacobian_matrix Stores the entries of the jacobian matrix as list of lists. Every entry of the matrix is
                             an expression that can evaluated in the `jacobian` function.
    """
    global state_variables
    global ode_definitions
    global function_variables
    global function_variable_definitions

    # defines the implementation of the jacobian matrix computation as an jinja2 template
    # Logic:
    # 1) for all function variables create a SymPy variable that stores its definition as a SymPy-expression
    # 2) For every ODE: create a SymPy symbol for every lefthandside of the ode and create a SymPy-expression
    # 3) Create the jacobian matrix as:
    #     every row is composed of the derivation of the righthandside of one with respect to every state variable from
    #     `state_variables`
    #     repeat this procedure for every ODE from `ode_definitions`
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
        "result_matrix_str = ''\n"
        "for rhs in right_hand_sides_expr:\n"
        "    row = []\n"
        "    for state_variable in state_variable_expr:\n"
        "        result_matrix_str += str(diff(rhs, state_variable)) + ' '\n"
        "        row.append(replace_state_variables_through_array_access(str(diff(rhs, state_variable))))\n"
        "    result_matrix_str = result_matrix_str + '\\n'\n"
        "    result_matrix.append(row)\n" )
    # create jinja2 template
    jacobian_function_implementation = jinja2.Template(jacobian_function_body)
    jacobian_function_implementation = jacobian_function_implementation.render(
        odes=zip(state_variables, ode_definitions),
        function_variables=zip(function_variables, function_variable_definitions))

    print "jacobian function implementation:"
    print jacobian_function_implementation
    # compile the generated code from the  jinja2 template for the better performance
    jacobian_function_implementation = compile(jacobian_function_implementation, '<string>', 'exec')
    # this matrix which is stored in the global variable is used in the `jacobian` function
    global jacobian_matrix
    jacobian_matrix = calculate_jacobian(jacobian_function_implementation)


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


def prepare_step_function():
    """
    Create the implementation of the  step function used in the GSL framework to compute one integration step.
    :param: global state_variables List of state variables which are passed as global variables
    :param: global ode_definitions List of  ode definitions for every state variable from `state_variables` which are 
                                   passed as global variables
    :param: global function_variables List of state variables which are passed as global variables
    :param: global function_variable_definitions List of  ode definitions for every state variable from 
                                                 `function_variables` which are passed as global variables

    :return: step_function_implementation Stores a compiled Python-code that performs one integration step. 
    """
    global state_variables
    global ode_definitions
    global function_variables
    global function_variable_definitions

    # The input provided as the fucntion parameters is not valid Python/GSL code. Convert all references of f_n and y_n
    # to f[n] and y[n]
    for idx in range(0, len(ode_definitions)):
        ode_definitions[idx] = replace_state_variables_through_array_access(ode_definitions[idx])
    for idx in range(0, len(function_variable_definitions)):
        function_variable_definitions[idx] = replace_state_variables_through_array_access(
            function_variable_definitions[idx])

    # defines the implementation of the jacobian matrix computation as an jinja2 template
    # Logic:
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


def replace_state_variables_through_array_access(definition):
    """
    Convert all references of f_n and y_n in `definition` to f[n] and y[n]
    :param definition: String to convert 
    :return: Converted string
    """
    match_f_or_y = re.compile(r"\b(y|f)_(\d)+\b")
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


def generate_representative_spike_train(sim_time_in_sec, slot_width_in_sec):
    """
    The function computes representative spike trains for the given simulation length. Uses a poisson distribution to 
    create biologically realistic characteristics of the spiketrains
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
    # spike_train = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    # return spike_train * time_slots


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
    """
    This function computes the average step size and the minimal step size that a given 
    integration method from GSL uses to evolve a certain system of ODEs during a certain
    simulation time, integration method from GSL and spike train for a given maximal stepsize.
    :param h: The maximal stepsize for one evolution step in miliseconds
    :param sim_time: The maximal total time of the simulation in miliseconds 
    :param integrator: A method from the GSL library for evolving ODES, e.g. rk4
    :param step_function: The function f that defines the system of ODEs as f(y)= y' that is evolved by the GSL method
    :pram jacobian: the jacobian matrix of the function `step_function`
    :param spikes: A representative spike train for the given simulation time `sim_time`
    :param y: The 'state variables' in f(y)=y'
    :return: Average and minimal step size.
    """
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


def draw_decision(step_min_imp, step_min_exp, step_average_imp, step_average_exp):
    """
    This function takes the 
    :param step_min_imp: 
    :param step_min_exp: 
    :param step_average_imp:
    :param step_average_exp:
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
        if step_average_imp > 2*step_average_exp:
            return "implicit"
        else:
            return "explicit"
    else:
        None  # This case cannot happen.

def step(t, y, params):
    """
    The GSL callback function that computes of integration step.
    :param y: the current state vector of the ODE system
    :param step_function_implementation: global variable that contains the implementation of the step function.
    :param _: Prescribed GSL parameters which are  not used in this step function.
    :return: Updated state vector stored in `f`
    """
    global step_function_implementation
    f = Numeric.zeros((dimension,), Numeric.Float)
    exec(step_function_implementation)
    return f


def jacobian(t, y, params):
    """
    Callback function that computes the jacobian matrix for the current state vector `y`.
    :param y: represents the current state vector of the ODE system as updated through the GSL integrator.
    :param _: another prescribed parameters which are not used here
    :return: dfdy that contains the jacobian matrix with respect to y`dfdt` is not computed now.
    """
    dfdy = Numeric.zeros((dimension, dimension), Numeric.Float)
    dfdt = Numeric.zeros((dimension,))

    # since the expression is put into a lambda function being evaluated it cannot access function's parameter
    # therefore, add the `y` variable to the globals manually
    g = globals()
    g["y"] = y

    # evaluate every entry of the `jacobian_matrix` and store the result in the corresponding entry of the `dfdy`
    for row in range(0, dimension):
        for col in range(0, dimension):
            # wrap the expression to a lambda function for the improved performance
            tmp = eval("lambda: " + jacobian_matrix[row][col], g)
            dfdy[row, col] = tmp()

    return dfdy, dfdt


def threshold(y, threshold_body):
    """
    :param y: represents the current state vector as updates through GSL.
    Evaluates the user defined threshold expression. 
    :param threshold_body: A python boolean expression that implements the threshold check.
    :return: true iff. the state vector `y` (or any entry) ceosses the threshold
    """
    # threshold is defined in terms of state variables. replace them through array accesses to `y` with corresponding
    # indices.
    threshold_body = replace_state_variables_through_array_access(threshold_body)
    return eval(threshold_body)
