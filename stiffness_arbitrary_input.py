import Numeric
import jinja2
import math
import pygsl.odeiv as odeiv
from sympy.parsing.sympy_parser import parse_expr
from sympy import *
import re

# the following variables must be defined globally since they are accessed from the step, jacobian, threshold and
# reset functions
# these functions are called from the framework and cannot take these variables as parameters
state_variables = []
right_hand_sides = []

# This declration is done here only due to performance considerations
step_function_body = (
    "{% for ode_var, ode in odes %}\n"
    "{{ode_var}} = {{ode}}\n"
    "{% endfor %}\n")


def replace_state_variables_through_array_access(definition):
    match_f_or_y = re.compile(r"\b(y|f)_(\d)+\b")
    result = match_f_or_y.sub(r"\1[\2]", definition)
    return result


def state_variables_to_f(state_varialbes):
    result = []
    for idx in range(0, len(state_varialbes)):
        result.append("f[" + str(idx) + "]")
    return result


def step(t, y, params):
    f = Numeric.zeros((dimension,), Numeric.Float)
    exec(step_function_implementation)
    return f


def threshold(y, threshold_body):
    threshold_body = replace_state_variables_through_array_access(threshold_body)
    return eval(threshold_body)


def reset(y, reset_statement):
    reset_statement = replace_state_variables_through_array_access(reset_statement)
    exec(reset_statement)


def generate_spike_train(SIMULATION_SLOTS):
    spike_train = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    return spike_train * int(math.ceil(float(SIMULATION_SLOTS) / len(spike_train)))


def evaluate_integrator(h,
                        sim_time,
                        simulation_slots,
                        integrator,
                        step_function,
                        jacobian,
                        spikes,
                        y,
                        initial_values,
                        threshold_body,
                        reset_statement):
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
            print str(time_slot) + ":   t=%f, current stepsize=%f y=" % (t, t - t_old), y
        s_min = s_min_old
        print "End while loop"

        if threshold is not None and threshold(y, threshold_body):
            reset(y, reset_statement)
            print("crossed")

        for idx, initial_value in enumerate(initial_values):
            y[idx] += initial_value * spikes[idx][time_slot]
    step_average = sim_time / step_counter
    return s_min_old, step_average


def calculate_jacobian(state_variables, right_hand_sides):
    state_variable_expr = []
    for state_variable in state_variables:
        state_variable_expr.append(parse_expr(state_variable))

    right_hand_sides_expr = []
    for ode in right_hand_sides:
        right_hand_sides_expr.append(parse_expr(ode))

    result_matrix = []
    for state_variable in state_variable_expr:
        row = []
        debug = ""
        for rhs in right_hand_sides_expr:
            debug += str(diff(rhs, state_variable)) + " "
            row.append(replace_state_variables_through_array_access(str(diff(rhs, state_variable))))
        print(debug)
        result_matrix.append(row)

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


def check_ode_system_for_stiffness(odes, default_values, threshold_body, reset_statement):
    global state_variables
    global right_hand_sides

    state_variables = []
    right_hand_sides = []

    # `default_values` contains a series of variables from the differential equations declaration separated by the
    # newline they are defined in the global scope so that different functions can access them either
    for default_value in default_values:
        exec(default_value) in globals()

    for ode in odes:
        lhs = ode.split("=")[0].strip().replace("f", "y")
        rhs = ode.split("=")[1].strip()

        state_variables.append(lhs)
        right_hand_sides.append(rhs)

    print("Starts stiffness test for the ODE system...")

    # this matrix is used in the jacobian function
    global jacobian_matrix
    jacobian_matrix = calculate_jacobian(state_variables, right_hand_sides)

    global dimension
    dimension = len(right_hand_sides)

    for idx in range(0, len(right_hand_sides)):
        right_hand_sides[idx] = replace_state_variables_through_array_access(right_hand_sides[idx])

    # This computation is done here only due to performance considerations
    global step_function_implementation
    step_function_implementation = jinja2.Template(step_function_body)
    step_function_implementation = step_function_implementation.render(
        odes=zip(state_variables_to_f(state_variables), right_hand_sides))

    # compile the code to boost the performance
    step_function_implementation = compile(step_function_implementation, '<string>', 'exec')

    h = 0.2
    sim_time = 200.
    simulation_slots = int(round(sim_time / h))
    gen_inh = generate_spike_train(simulation_slots)
    print("####SUMMARY####")
    print("State variables:\n" + str(state_variables))
    print("ODE definitions:")
    for ode in right_hand_sides:
        print(ode)
    print("####END####")
    print ("######### rk imp #########")
    step_min_imp, step_average_imp = evaluate_integrator(
        h,
        sim_time,
        simulation_slots,
        odeiv.step_bsimp,
        step,
        jacobian,
        [gen_inh] * dimension,
        start_values,
        initial_values,
        threshold_body,
        reset_statement)

    # `default_values` contains a series of variables from the differential equations declaration separated by the
    # newline they are defined in the global scope so that different functions can access them either
    for default_value in default_values:
        exec (default_value) in globals()
    print ("######### rk expl ########")
    step_min_exp, step_average_exp = evaluate_integrator(
        h,
        sim_time,
        simulation_slots,
        odeiv.step_rk4,
        step,
        jacobian,
        [gen_inh] * dimension,
        start_values,
        initial_values,
        threshold_body,
        reset_statement)

    print ("######### results #######")
    print "min_imp: {} min_exp: {}".format(step_min_imp, step_min_exp)
    print "avg_imp: {} avg_exp: {}".format(step_average_imp, step_average_exp)
    print ("########## end ##########")
