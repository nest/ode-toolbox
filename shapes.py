"""Classes for storing and processing post-synaptic shapes.

This file is part of the NEST ODE toolbox. It provides two classes:
`ShapeFunction` calculates and stores the properties of a
post-synaptic shape given as a function of time, `ShapeODE` stores the
properties of a shape given in the form of an ordinary differential
equation (ODE).

The classes are used for the construction of an appropriate evolution
scheme for ODEs that contain these shapes. In a neuron model, a shape
is convolved with the spike input in order to calculate the
post-synaptic response.

"""


from sympy.parsing.sympy_parser import parse_expr
from sympy import *
from sympy.matrices import zeros


class ShapeFunction(object):
    """Class for representing a shape given as a function of time.

    `ShapeFunction` is constructed from the name of a shape and its
    mathematical description. During initialization, the class checks
    if the shape satisfies a linear homogeneous ODE. If the check
    succeeds, the class will provide the properties `name`, `order`,
    `nestml_ode_form`, `derivative_factors` and `intitial_values`.

    Example:
    ========

    shape_alpha = ShapeFunction("shape_alpha", "e / tau * t * exp(-t / tau)")

    * `name` will be set to the SymPy symbol "shape_alpha".
    * `order` is set to 2, because the given function satisfies a
      homogeneos ODE of order 2.
    * `nestml_ode_form` will be set to the linear homogeneous ODE that
      the given function satisfies. In the example it will be set to
      "shape_alpha'' = -1 / tau**2 * shape_alpha -2 / tau * shape_alpha'
    * `derivative_factors` will be a set to the list of factors that
      occur in the calculated ODE (`nest_ode_form`). In the example
      the list will be [-1/tau**2, -2/tau]
    * `initial_values` will be set to [0, e/tau] as these are the
      initial values of all derivatives in the ODE from lowest to
      highest derivative.

    """

    # The algorithm requires some pre-defined symbols. `t` represents
    # time, `derivative_factor` is the factor in front of the shape in
    # "shape' = derivative_factor * shape" in case shape satisfies
    #  a linear homogeneous ODE.
    derivative_factor, t = symbols("derivative_factor, t")


    def __init__(self, name, function_def):

        # Convert the name of the shape function (given as a string)
        # to a SymPy symbol
        self.name = parse_expr(name)

        # Convert the shape function (given as a string) to its
        # symbolic representation
        self.shape_expr = parse_expr(function_def)

        # `found_ode` is set to `True` if we find a linear homogeneous
        # ODE that the given shape satisfies
        found_ode = False

        # First we check if `shape` satisfies a linear homogeneous ODE
        # of order 1.
        order = 1

        # `derivatives` is a list of all derivatives of `shape` up to
        # the order we are checking, starting at 0.
        derivatives = [self.shape_expr, diff(self.shape_expr, self.t)]

        t_val = None
        max_t = 100

        # The algorithm uses a matrix whose entries correspond to the
        # evaluation of derivatives of the shape function at certain
        # points in time (`t`). In some (unlikely) cases the matrix is
        # not invertible for a given `t` and we need to check other
        # `t`s to make sure we are not dividing by zero.
        for k in range(1, max_t):
            if derivatives[0].subs(self.t, k) != 0:
                t_val = k
                break

        if t_val is None:
            raise Exception('No suitable t found after {} tries.'.format(max_t))

        # `derivative_factors` is the list of the potential factors in
        # the ODE from the factor of shape^(0) to shape^(order-1). It's
        # length corresponds to the order of the ODE.
        derivative_factors = [(1 / derivatives[0] * derivatives[1]).subs(self.t, t_val)]

        # `diff_rhs_lhs` is the difference of the derivative of shape
        # of order 'order' and the sum of all lower derivatives times
        # their 'derivative_factors'.
        diff_rhs_lhs = derivatives[order] - derivative_factors[order - 1] * derivatives[order - 1]

        # If `diff_rhs_lhs` equals 0 for some `derivative_factors`,
        # `shape` satisfies a first order linear homogeneous ODE (and
        # does for t=1).
        if simplify(diff_rhs_lhs) == sympify(0):
            found_ode = True

        # While an ODE has not yet been found and we have not yet
        # reached the maximum order we are checking for, we check if
        # `shape` satisfies a linear homogeneous ODE of the next
        # higher order.
        max_order = 10
        while not found_ode and order < max_order:

            # The potential order must be at least `order+1`
            order += 1

            # Add the next higher derivative to the list of
            # derivatives of `shape`
            derivatives.append(diff(derivatives[-1], self.t))

            # The goal here is to calculate the factors (`derivative_factors`)
            # of the ODE (assuming they exist).

            # The idea is to create a system of equations by
            # substituting natural numbers into the homogeneous linear
            # ODE with variable derivative factors order many times
            # for varying natural numbers and solving for derivative
            # factors. Once we have derivative factors, the ODE is
            # uniquely defined. This is assuming that shape satisfies
            # an ODE of this order, which we check after determining
            # the factors.

            # `X` will contain derivatives up to `order-1` of some
            # natural numbers as rows (differing in each row)
            X = zeros(order)

            # `Y` will contain the derivatives of `order` of the natural
            # number in the corresponding row of `X`
            Y = zeros(order, 1)

            # It is possible that by choosing certain natural numbers,
            # the system of equations will not be solvable, i.e. `X` is
            # not invertible. This is unlikely but we check for
            # invertibility of `X` for varying sets of natural numbers.
            invertible = False
            for k in range(max_t):
                for i in range(order):
                    substitute = i + k + 1
                    Y[i] = derivatives[order].subs(self.t, substitute)
                    for j in range(order):
                        X[i, j] = derivatives[j].subs(self.t, substitute)
                if det(X) != 0:
                    invertible = True
                    break

            if not invertible:
                raise Exception("Failed to find a homogeneous linear ODE "
                                "which shape satisfies, or shape does not "
                                "satisfy any such ODE of order <= {}".format(max_order))


            # If the `order`th derivative of the shape equals the C_i
            # times the `i`th derivative of the shape (C_i are the
            # derivative factors), we can find the derivative factors
            # for `order-1` by evaluating the previous equation as a
            # linear system of order `order-1` such that Y =
            # [shape^order] = X * [C_i]. Hence [C_i] can be found by
            # inverting X.
            derivative_factors = X.inv() * Y
            diff_rhs_lhs = 0

            # We calculated the `derivative_factors` of the linear
            # homogeneous ODE of order `order` and only assumed that
            # shape satisfies such an ODE. We now have to check that
            # this is actually the case:
            for k in range(order):
                # sum up derivatives 'shapes' times their potential 'derivative_factors'
                diff_rhs_lhs -= derivative_factors[k] * derivatives[k]
            diff_rhs_lhs += derivatives[order]

            if simplify(diff_rhs_lhs) == sympify(0):
                found_ode = True
                break

        if not found_ode:
            raise Exception("Shape does not satisfy any ODE of order <= {}".format(max_order))

        self.order = order
        self.derivative_factors = list(simplify(derivative_factors))
        self.initial_values = [x.subs(self.t, 0) for x in derivatives[:-1]]
        self.updates_to_state_shape_variables = []  # must be filled after the propagator matrix is computed

        self.nestml_ode_form = []
        for cur_order in range(0, order-1):
            if cur_order > 0:
                self.nestml_ode_form.append({name + "__" + str(cur_order): name + "__" + str(cur_order + 1)})
            else:
                self.nestml_ode_form.append({name: name + "__1"})

        # Compute the right and left hand side of the ODE that 'shape' satisfies
        rhs_str = []
        for k in range(order):
            if k > 0:
                rhs_str.append("{} * {}__{}".format(simplify(derivative_factors[k]), name, str(k)))

            else:
                rhs_str.append("{} * {}".format(simplify(derivative_factors[k]), name))

        rhs = " + ".join(rhs_str)
        if order == 1:
            lhs = name
        else:
            lhs = name + "__" + str(order-1)

        self.nestml_ode_form.append({lhs: rhs})

    def additional_shape_state_variables(self):
        """

        :return: Creates list with state shapes variables in the `reversed` order, e.g. [I'', I', I]
        """
        result = []
        for order in range(0, self.order):
            if order > 0:
                result = [(str(self.name) + "__" + str(order))] + result
            else:
                result = [str(self.name)] + result
        return result

    def add_update_to_shape_state_variable(self, shape_state_variable, shape_state_variable_update):
        self.updates_to_state_shape_variables = [{str(shape_state_variable): str(shape_state_variable_update)}] + self.updates_to_state_shape_variables

    def get_updates_to_shape_state_variables(self):
        result = []
        if self.order > 0:  # FIX ME

            for entry_map in self.updates_to_state_shape_variables:
                # by construction, there is only one value in the `entry_map`
                for shape_state_variable, shape_state_variable_update in entry_map.iteritems():
                    result.append({"__tmp__" + shape_state_variable: shape_state_variable_update})

            for entry_map in self.updates_to_state_shape_variables:
                # by construction, there is only one value in the `entry_map`
                for shape_state_variable, shape_state_variable_update in entry_map.iteritems():
                    result.append({shape_state_variable: "__tmp__" + shape_state_variable})

        else:
            result = self.updates_to_state_shape_variables

        return result

    def get_initial_values(self):
        result = []
        for idx, initial_value in enumerate(self.initial_values):
            if idx > 0:
                p = {"iv__" + str(self.name) + "__" + str(idx): str(initial_value)}
            else:
                p = {"iv__" + str(self.name): str(initial_value)}
            result = [p] + result
        return result


class ShapeODE(object):
    """
    Provides a class 'ShapeODE'. An instance of `ShapeODE` is
    defined with the name of the shape (i.e a function of `t`
    that satisfies a certain ODE), the variables on the left handside
    of the ODE system, the right handsides of the ODE systems
    and the initial value of the function.

    Equations are of the form I''' = a*I'' + b*I' + c*I

    Example:
    ========

    ShapeODE("shape_alpha",   # name/variable
             "-1/tau**2 * shape_alpha -2/tau * shape_alpha'", rhs of ODE
             ["0", "e/tau"]) # initial values

    Canonical calculation of the properties, `order`, `name`,
    `initial_values` and the system of ODEs in matrix form are made.
    """
    def __init__(self, name, ode_sys_rhs, initial_values):

        self.name = parse_expr(name)
        self.order = len(initial_values)
        self.initial_values = [parse_expr(i) for i in initial_values]

        self.ode_sys_vars = [symbols(name+"d"*i) for i in range(self.order) ]
        self.ode_sys_rhs = parse_expr(ode_sys_rhs.replace("'", "d"))

        print self.ode_sys_vars
        print self.ode_sys_rhs

        self.matrix = zeros(self.order)

        derivative_factors = []
        for ode_sys_var in self.ode_sys_vars:
            derivative_factors.append(diff(self.ode_sys_rhs, ode_sys_var))

        print "derivative_factors:", derivative_factors
