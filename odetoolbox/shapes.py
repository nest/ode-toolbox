#
# shapes.py
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


"""Components of the NEST ODE toolbox for storing and processing
post-synaptic shapes.

"""

import re
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy import det, diff, Symbol, sympify, simplify
from sympy.matrices import zeros


def is_sympy_type(var):
    return isinstance(var, tuple(sympy.core.all_classes))


class Shape(object):
    """Canonical representation of a postsynaptic shape.

    Description
    -----------

    This class provides a canonical representation of a postsynaptic shape independently of the way in which the user specified the shape. It assumes a differential equation of the general form (where bracketed superscript :math:`\cdot^{(n)}` indicates the n-th derivative with respect to time):

    .. math::

        x^{(n)} = N + \sum_{i=0}^{n-1} c_i x^{(i)}

    Any constant or nonlinear part is here contained in the term N.

    In the input and output, derivatives are indicated by adding one prime (single quotation mark) for each derivative order. For example, in the expression

    .. code::

        I''' = c0 I + c1 I' + c2 I'' + N

    the `symbol` of the ODE would be `I` (i.e. without any qualifiers), `order` would be 3, and `derivative_factors` would be {"I" : "f0", "I'" : "f1", "I''" : f2 }

    Internally, the representation is based on the following attributes:

    Attributes
    ----------

    symbol : SymPy expression
        Symbolic name of the shape without additional qualifiers like prime symbols or similar.
    order : int
        Order of the ODE representing the shape.
    initial_values : dict of SymPy expressions
        Initial values of the ODE representing the shape. The dict contains `order` many key-value pairs: one for each derivative that occurs in the ODE. The keys are strings created by concatenating the variable symbol with as many single quotation marks (') as the derivation order. The values are SymPy expressions.
    derivative_factors : list of SymPy expressions
        The factors for the derivatives that occur in the ODE. This list has to contain `order` many values, i.e. one for each derivative that occurs in the ODE. The values have to be in ascending order, i.e. a0 df_d0, iv_d1, ... for the derivatives d0, d1, ...
    """

    def __init__(self, symbol, order, initial_values, derivative_factors, diff_rhs_derivatives=sympy.parsing.sympy_parser.parse_expr("0")):
        """Perform type and consistency checks and assign arguments to member variables."""
        assert type(symbol) is Symbol, "symbol is not a SymPy symbol: \"%r\"" % symbol
        self.symbol = symbol

        assert type(order) is int, "order is not an integer: \"%d\"" % order
        self.order = order

        assert len(initial_values) == order, "length of initial_values != order"
        for iv_name, iv in initial_values.items():
            assert is_sympy_type(iv), "initial value for %s is not a SymPy expression: \"%r\"" % (iv_name, iv)
        self.initial_values = initial_values

        assert len(derivative_factors) == order, "length of derivative_factors != order"
        for df in derivative_factors:
            assert is_sympy_type(df), "derivative factor is not a SymPy expression: \"%r\"" % iv
        self.derivative_factors = derivative_factors

        # Compute the state variables for ODE the shape satisfies
        self.state_variables = []
        for i in range(self.order):
            if i > 0:
                self.state_variables.insert(0, Symbol("{}{}".format(str(symbol), "__d" * i)))
            else:
                self.state_variables.insert(0, symbol)

        ## Compute the definition of the ODE the shape satisfies
        #rhs = ["{} * {}".format(simplify(derivative_factors[0]), symbol)]
        #for k in range(1, order):
            #rhs.append("{} * {}{}".format(simplify(derivative_factors[k]), symbol, "__d" * i))
        #self.ode_definition = " + ".join(rhs)
        #if not diff_rhs_derivatives == sympy.parsing.sympy_parser.parse_expr("0"):
            #self.ode_definition = " + " + str(diff_rhs_derivatives)

        self.diff_rhs_derivatives = diff_rhs_derivatives


    def __str__(self):
        s = "Shape \"" + str(self.symbol) + "\" of order " + str(self.order)
        return s


    def is_lin_const_coeff(self, shapes):
        """
        :return true iff the ode definition is a linear and constant coefficient ODE in all known variable symbols in `shapes`
        """
        print("checking lin const for " + str(self))
        all_symbols = []
        for shape in shapes:
            for order in range(shape.order):
                all_symbols.append(Symbol(str(shape.symbol) + "__d" * order))
        
        for sym in all_symbols:
            expr = diff(self.diff_rhs_derivatives, sym)
            if not sympify(expr) == sympify(0):
                # symbol `sym` appears on right-hand side of this expression. Check to see if it appears as a linear term by checking whether taking its derivative again yields 0
                return sympify(diff(expr, sym)) == sympify(0)
        return True
                
                
                
        #for derivative_factor in self.diff_rhs_derivatives:
        ## xxx: todo: Loop over all derivative factors; check if equal to sympy.numbers.Float or Integer
        #return self.diff_rhs_derivatives == sympy.parsing.sympy_parser.parse_expr("0")


    #def is_homogeneous_ode(self):
        #"""`diff_rhs_derivatives` contains the nonlinear and homogeneous part; check if it contains a nonlinear part"""
        #all_shape_symbols = [ Symbol(self.symbol+"__d"*i) for i in range(self.order) ]
        #for derivative_symbol in all_shape_symbols:
            #if not diff(self.diff_rhs_derivatives, derivative_symbol) == sympify(0):
                #return False
        #return True
        ##return type(sympify(self.diff_rhs_derivatives).n())


    @classmethod
    def from_json(cls, indict):
        """Create a shape object from a JSON input dictionary
        """

        if not "expression" in indict:
            raise Exception("No `expression` keyword found in input")

        lhs_match = re.search(".*=", indict["expression"])
        if lhs_match is None:
            raise Exception("Error while parsing expression \"" + indict["expression"] + "\"")
        lhs = lhs_match.group()[:-1]

        rhs_match = re.search("=.*", indict["expression"])
        if rhs_match is None:
            raise Exception("Error while parsing expression \"" + indict["expression"] + "\"")
        rhs = rhs_match.group()[1:]

        symbol_match = re.search("[a-zA-Z_][a-zA-Z0-9_]*", lhs)
        if symbol_match is None:
            raise Exception("Error while parsing symbol name in \"" + lhs + "\"")
        symbol = symbol_match.group()
        order = len(re.findall("'", lhs))

        initial_values = {}
        if not "initial_value" in indict.keys() \
         and not "initial_values" in indict.keys() \
         and order > 0:
            raise Exception("No initial values specified for order " + str(order) + " equation with variable symbol \"" + symbol + "\"")

        if "initial_value" in indict.keys() \
         and "initial_values" in indict.keys():
            raise Exception("`initial_value` and `initial_values` cannot be specified simultaneously for equation with variable symbol \"" + symbol + "\"")

        if "initial_value" in indict.keys():
            if not order == 1:
                raise Exception("Single initial value specified for equation that is not first order in equation with variable symbol \"" + symbol + "\"")
            initial_values[symbol] = indict["initial_value"]

        if "initial_values" in indict.keys():
            if not len(indict["initial_values"]) == order:
                raise Exception("Wrong number of initial values specified for order " + str(order) + " equation with variable symbol \"" + symbol + "\"")

            for iv_lhs, iv_rhs in indict["initial_values"].items():
                symbol_match = re.search("[a-zA-Z_][a-zA-Z0-9_]*", iv_lhs)
                if symbol_match is None:
                    raise Exception("Error trying to parse initial value variable symbol from string \"" + iv_lhs + "\"")
                iv_symbol = symbol_match.group()
                if not iv_symbol == symbol:
                    raise Exception("Initial value variable symbol \"" + iv_symbol + "\" does not match equation variable symbol \"" + symbol + "\"")
                iv_order = len(re.findall("'", iv_lhs))
                initial_values[iv_symbol + iv_order * "'"] = iv_rhs

        if order == 0:
            return Shape.from_function(symbol, rhs)
        else:
            return Shape.from_ode(symbol, rhs, initial_values)

    @classmethod
    def from_function(cls, symbol, definition, max_t=100, max_order=10):
        """Create a Shape object given a function of time.

        The goal of the algorithm is to calculate the factors of the ODE,
        assuming they exist. It uses a matrix whose entries correspond to
        the evaluation of derivatives of the shape function at certain
        points `t` in time.

        The idea is to create a system of equations by substituting
        natural numbers into the homogeneous linear ODE with variable
        derivative factors order many times for varying natural numbers
        and solving for derivative factors. Once we have derivative
        factors, the ODE is uniquely defined. This is assuming that shape
        satisfies an ODE of this order, which we check after determining
        the factors.

        In the function, the symbol `t` is assumed to stand for the
        current time.

        The algorithm used in this function is described in full detail
        together with the mathematical foundations in the following
        publication:

            Inga Blundell, Dimitri Plotnikov, Jochen Martin Eppler,
            Abigail Morrison (2018) Automatically selecting an optimal
            integration scheme for systems of differential equations in
            neuron models. Front. Neuroinf. doi:10.3389/fninf.2018.00050.

        Parameters
        ----------
        symbol : string
            The symbol of the shape (e.g. "alpha", "I", "exp")
        definition : string
            The definition of the shape (e.g. "(e/tau_syn_in) * t *
            exp(-t/tau_syn_in)")

        Returns
        -------
        shape : Shape
            The canonical representation of the postsynaptic shape

        Examples
        --------
        >>> Shape("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        """

        # Create a SymPy symbols the time (`t`)
        t = Symbol("t")

        # The symbol and the definition of the shape function were given as
        # strings. We have to transform them to SymPy symbols for using
        # them in symbolic calculations.
        symbol = parse_expr(symbol)
        shape = parse_expr(definition)

        # `derivatives` is a list of all derivatives of `shape` up to the
        # order we are checking, starting at 0.
        derivatives = [shape, diff(shape, t)]

        # We first check if `shape` satisfies satisfies a linear
        # homogeneous ODE of order 1.
        order = 1

        # To avoid a division by zero below, we have to find a `t` so that
        # the shape function is not zero at this `t`.
        t_val = None
        for t_ in range(1, max_t):
            if derivatives[0].subs(t, t_) != 0:
                t_val = t_
                break

        if t_val is None:
            # It is very unlikely that the shape obeys a linear homogeneous
            # ODE of order 1 and we still did not find a suitable
            # `t_val`. This would mean that the function evaluates to zero at
            # `t_` = 1, ..., `max_t`, which usually hints at an error in the
            # specification of the function.
            msg = "Cannot find t for which shape function is unequal to zero"
            raise Exception(msg)

        # `derivative_factors` contains the factor in front of the
        # derivative in the ODE that the shape function potentially
        # obeys.  This is a list just for consistency with later
        # cases, where multiple factors are calculatedrequired
        derivative_factors = [(1 / derivatives[0] * derivatives[1]).subs(t, t_val)]

        # `diff_rhs_lhs` is the difference between the derivative of
        # shape and the shape itself times its derivative factor.
        diff_rhs_lhs = derivatives[1] - derivative_factors[0] * derivatives[0]

        # If `diff_rhs_lhs` equals 0, `shape` satisfies a first order
        # linear homogeneous ODE. We set the flag `found_ode`
        # correspondingly.
        found_ode = simplify(diff_rhs_lhs) == sympify(0)

        # If `shape` does not satisfy a linear homogeneous ODE of order 1,
        # we try to find one of higher order in a loop. The loop runs
        # while no linear homogeneous ODE was found and the maximum
        # order to check for was not yet reached.
        while not found_ode and order < max_order:

            # Set the potential order for the iteration
            order += 1

            # Add the next higher derivative to the list
            derivatives.append(diff(derivatives[-1], t))

            # `X` is an `order`x`order` matrix that will be assigned
            # the derivatives up to `order`-1 of some natural numbers
            # as rows (differing in each row)
            X = zeros(order)
        
            # `Y` is a vector of length `order` that will be assigned
            # the derivatives of `order` of the natural number in the
            # corresponding row of `X`
            Y = zeros(order, 1)
        
            # It is possible that by choosing certain natural numbers,
            # the system of equations will not be solvable, i.e. `X`
            # is not invertible. This is unlikely but we check for
            # invertibility of `X` for varying sets of natural
            # numbers.
            invertible = False
            for t_ in range(max_t):
                for i in range(order):
                    substitute = i + t_ + 1
                    Y[i] = derivatives[order].subs(t, substitute)
                    for j in range(order):
                        X[i, j] = derivatives[j].subs(t, substitute)
        
                if det(X) != 0:
                    invertible = True
                    break

            # If we failed to find an invertible `X` for `t` = 1, ...,
            # `max_t`, it is very unlikely that the shape function
            # obeys a linear homogeneous ODE of order `order` and we
            # go on checking the next potential order. Else we
            # calculate the derivative factors and check if shape
            # actually obeys the corresponding ODE.
            if invertible:
            
                # The `order`th derivative of the shape equals `C_i`
                # times the `i`th derivative of the shape (`C_i` are
                # the derivative factors). This means we can find the
                # derivative factors for `order-1` by evaluating the
                # previous equation as a linear system of order
                # `order-1` such that Y = [shape^order] = X *
                # [C_i]. Hence [C_i] can be found by inverting X.
                derivative_factors = X.inv() * Y
                diff_rhs_lhs = 0

                # We calculated the `derivative_factors` of the linear
                # homogeneous ODE of order `order` and only assumed
                # that shape satisfies such an ODE. We now have to
                # check that this is actually the case:
                for k in range(order):
                    diff_rhs_lhs -= derivative_factors[k] * derivatives[k]
                    #if not diff(derivative_factors[k], derivatives[k]) == sympify(0):
                        #raise Exception("caboodle")
                diff_rhs_lhs += derivatives[order]

                if simplify(diff_rhs_lhs) == sympify(0):
                    found_ode = True
                    break

        if not found_ode:
            msg = "Shape does not satisfy any ODE of order <= " % max_order
            raise Exception(msg)

        # Calculate the initial values of the found ODE and simplify the
        # derivative factors before creating and returning the Shape
        # object.
        initial_values = { str(symbol) + derivative_order * '\'' : x.subs(t, 0) for derivative_order, x in enumerate(derivatives[:-1]) }
        derivative_factors = [simplify(df) for df in derivative_factors]
        return cls(symbol, order, initial_values, derivative_factors)

    @classmethod
    def from_ode(cls, symbol, definition, initial_values, **kwargs):
        """Create a Shape object given an ODE and initial values.

        Provides a class 'ShapeODE'. An instance of `ShapeODE` is
        defined with the symbol of the shape (i.e a function of `t`
        that satisfies a certain ODE), the variables on the left hand side
        of the ODE system, the right hand sides of the ODE systems
        and the initial value of the function.

        Equations are of the form I''' = a*I'' + b*I' + c*I

        Canonical calculation of the properties, `order`, `symbol`,
        `initial_values` and the system of ODEs in matrix form are made.

        Parameters
        ----------
        symbol : string
            The symbol (variable name) of the ODE
        definition : string
            The definition of the ODE
        initial_values : dict
            A dictionary mapping initial values to expressions.

        Examples
        --------
        Shape.from_ode("alpha",
                       "-1/tau**2 * shape_alpha -2/tau * shape_alpha'",
                       {"alpha" : "0", "alpha'" : "e/tau", "0"]})
        """

        order = len(initial_values)

        def _initial_values_sanity_checks():
            assert type(initial_values) is dict, "Initial values should be specified as a dictionary"
            _order_from_definition = 1
            _re_search = re.compile(symbol + "'+").findall(definition)

            for match in _re_search:
                __re_search = re.compile("'+").search(match)
                _order = 0
                if not __re_search is None:
                    _order = len(__re_search.group())
                _order_from_definition = max(_order + 1, _order_from_definition)

            assert _order_from_definition == order, "Wrong number of initial values specified, expected " + str(_order_from_definition) + ", got " + str(order)

            initial_val_specified = [False] * order
            for k, v in initial_values.items():
                if not k[0:len(symbol)] == symbol:
                    raise Exception("In definition for " + str(symbol) + ": Initial value specified for unknown variable symbol \"" + k + "\"")
                _order = 0
                _re_search = re.compile("'+").search(k)
                if not _re_search is None:
                    _order = len(_re_search.group())
                    if _order >= order:
                        raise Exception("In defintion of initial value for variable \"" + k + "\": differential order (" + str(_order) + ") exceeds that of overall equation order (" + str(order) + ")")
                    initial_val_specified[_order] = True
                else:
                    # _order == 0
                    if initial_val_specified[_order]:
                        raise Exception("Initial value for zero-th order specified more than once")
                    initial_val_specified[_order] = True

            if not all(initial_val_specified):
                raise Exception("Initial value not specified for all differential orders")

        _initial_values_sanity_checks()

        initial_values = { k : parse_expr(v) for k, v in initial_values.items() }
        derivative_symbols = [ Symbol(symbol+"__d"*i) for i in range(order) ]
        definition = parse_expr(definition.replace("'", "__d"))
        symbol = parse_expr(symbol)

        # the purely linear part of the shape goes in `derivative_factors`
        derivative_factors = []
        for derivative_symbol in derivative_symbols:
            diff_expr = diff(definition, derivative_symbol)
            if not diff(diff_expr, derivative_symbol) == sympify(0):
                # the equation is nonlinear in this `derivative_symbol`; do not put this term into `derivative_factors` but leave it for `diff_rhs_derivatives`
                derivative_factors.append(sympify(0))
            else:
                derivative_factors.append(diff_expr)

        # the nonlinear and non-homogeneous parts are in `diff_rhs_derivatives`
        diff_rhs_derivatives = definition
        for derivative_factor, derivative_symbol in zip(derivative_factors, derivative_symbols):
            diff_rhs_derivatives -= derivative_factor * derivative_symbol

        return cls(symbol, order, initial_values, derivative_factors, diff_rhs_derivatives)

