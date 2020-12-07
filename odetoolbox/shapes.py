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

import functools
import logging
import numpy as np
import re
import sympy
import sympy.parsing.sympy_parser

from .sympy_printer import _is_sympy_type


class MalformedInputException(Exception):
    """
    Thrown in case an error occurred while processing malformed input.
    """
    pass


class Shape():
    r"""
    This class provides a canonical representation of a shape function independently of the way in which the user specified the shape. It assumes a differential equation of the general form (where bracketed superscript :math:`{}^{(n)}` indicates the :math:`n`-th derivative with respect to time):

    .. math::

       x^{(n)} = N + \sum_{i=0}^{n-1} c_i x^{(i)}

    Any constant or nonlinear part is here contained in the term N.

    In the input and output, derivatives are indicated by adding one prime (single quotation mark) for each derivative order. For example, in the expression

    ::

       x''' = c0*x + c1*x' + c2*x'' + x*y + x**2

    the :python:`symbol` of the ODE would be :python:`x` (i.e. without any qualifiers), :python:`order` would be 3, :python:`derivative_factors` would contain the linear part in the form of the list :python:`[c0, c1, c2]`, and the nonlinear part :python:`x*y + x**2` is stored in :python:`diff_rhs_derivatives`.
    """

    EXPRESSION_SIMPLIFICATION_THRESHOLD = 1000

    # a minimal subset of sympy classes and functions to avoid "from sympy import *"
    _sympy_globals = {"Symbol": sympy.Symbol,
                      "Integer": sympy.Integer,
                      "Float": sympy.Float,
                      "Function": sympy.Function,
                      "Pow": sympy.Pow,
                      "power": sympy.Pow,
                      "exp": sympy.exp,
                      "log": sympy.log,
                      "sin": sympy.sin,
                      "cos": sympy.cos,
                      "tan": sympy.tan,
                      "asin": sympy.asin,
                      "sinh": sympy.sinh,
                      "asinh": sympy.asinh,
                      "acos": sympy.acos,
                      "cosh": sympy.cosh,
                      "acosh": sympy.acosh,
                      "tanh": sympy.tanh,
                      "atanh": sympy.atanh,
                      "e": sympy.exp(1),
                      "E": sympy.exp(1),
                      "t": sympy.Symbol("t"),
                      "DiracDelta": sympy.DiracDelta}


    def __init__(self, symbol, order, initial_values, derivative_factors, diff_rhs_derivatives=sympy.Float(0.), lower_bound=None, upper_bound=None, derivative_symbol="__d", debug=False):
        r"""
        Perform type and consistency checks and assign arguments to member variables.

        :param symbol: Symbolic name of the shape without additional qualifiers like prime symbols or similar.
        :param order: Order of the ODE representing the shape.
        :param initial_values: Initial values of the ODE representing the shape. The dict contains :python:`order` many key-value pairs: one for each derivative that occurs in the ODE. The keys are strings created by concatenating the variable symbol with as many single quotation marks (') as the derivation order. The values are SymPy expressions.
        :param derivative_factors: The factors for the derivatives that occur in the ODE. This list has to contain :path:`order` many values, i.e. one for each derivative that occurs in the ODE. The values have to be in ascending order, i.e. :python:`[c0, c1, c2]` for the given example.
        :param diff_rhs_derivatives: Nonlinear part of the ODE representing the shape, i.e. :python:`x*y + x**2` for the given example.
        """
        if not type(symbol) is sympy.Symbol:
            raise MalformedInputException("symbol is not a SymPy symbol: \"%r\"" % symbol)

        self.symbol = symbol

        if str(symbol) in Shape._sympy_globals.keys():
            raise MalformedInputException("The symbol name " + str(symbol) + " clashes with a predefined symbol by the same name")

        if not type(order) is int:
            raise MalformedInputException("order should be of type int, but has type " + str(type(order)))

        self.order = order

        if not len(initial_values) == order:
            raise MalformedInputException(str(len(initial_values)) + " initial values specified, while " + str(order) + " were expected based on differential equation definition")
        for iv_name, iv in initial_values.items():
            if not _is_sympy_type(iv):
                raise MalformedInputException("initial value for %s is not a SymPy expression: \"%r\"" % (iv_name, iv))
            differential_order = iv_name.count("'")
            if differential_order > 0:
                iv_basename = iv_name[:-differential_order]
            else:
                iv_basename = iv_name
            if not iv_basename == str(symbol):
                raise MalformedInputException("Initial value specified for unknown variable \"" + str(iv_basename) + "\" in differential equation for variable \"" + str(symbol) + "\"")
            if differential_order >= self.order:
                raise MalformedInputException("Initial value \"" + str(iv_name) + "\" specifies initial value for degree " + str(differential_order) + ", which is too high in order-" + str(self.order) + " differential equation")

        self.initial_values = initial_values

        for sym in initial_values.keys():
            if derivative_symbol in str(sym):
                raise MalformedInputException("Input variable name \"" + str(sym) + "\" should not contain the string \"" + derivative_symbol + "\", which is used to indicate variable differential order")

        if not len(derivative_factors) == order:
            raise MalformedInputException(str(len(derivative_factors)) + " derivative factors specified, while " + str(order) + " were expected based on differential equation definition")

        for df in derivative_factors:
            if not _is_sympy_type(df):
                raise MalformedInputException("Derivative factor \"%r\" is not a SymPy expression" % df)

        self.derivative_factors = derivative_factors

        if len(str(diff_rhs_derivatives)) > Shape.EXPRESSION_SIMPLIFICATION_THRESHOLD:
            logging.warning("Shape \"" + str(self.symbol) + "\" initialised with an expression that exceeds SymPy simplification threshold")
            self.diff_rhs_derivatives = diff_rhs_derivatives
        else:
            self.diff_rhs_derivatives = sympy.simplify(diff_rhs_derivatives)

        self.lower_bound = lower_bound
        if not self.lower_bound is None:
            self.lower_bound = sympy.simplify(self.lower_bound)

        self.upper_bound = upper_bound
        if not self.upper_bound is None:
            self.upper_bound = sympy.simplify(self.upper_bound)

        logging.debug("Created Shape with symbol " + str(self.symbol) + ", derivative_factors = " + str(self.derivative_factors) + ", diff_rhs_derivatives = " + str(self.diff_rhs_derivatives))


    def __str__(self):
        s = "Shape \"" + str(self.symbol) + "\" of order " + str(self.order)
        return s


    def is_homogeneous(self, shapes=None, differential_order_symbol="__d"):
        r"""
        :return: :python:`False` if and only if the shape has a nonzero right-hand side.
        :rtype: bool
        """

        if self.diff_rhs_derivatives.is_zero:
            # trivial case: right-hand side is zero
            return True

        all_symbols = self.get_all_variable_symbols(shapes, derivative_symbol=differential_order_symbol)
        for term in sympy.Add.make_args(self.diff_rhs_derivatives.expand()):
            term_is_const = True
            for sym in all_symbols:
                expr = sympy.diff(term, sym)
                if not sympy.sympify(expr).is_zero:
                    # this term is of the form "sym * expr", hence it cannot be a constant term
                    term_is_const = False

            if term_is_const:
                return False

        return True


    def get_initial_value(self, sym: str):
        r"""
        Get the initial value corresponding to the variable symbol.

        :param sym: String representation of a sympy symbol, e.g. :python:`"V_m'"`
        """
        if not sym in self.initial_values.keys():
            return None
        return self.initial_values[sym]


    def get_state_variables(self, derivative_symbol="'"):
        r"""
        Get all variable symbols for this shape, ordered according to derivative order: :python:`[sym, dsym/dt, d^2sym/dt^2, ...]`

        :return: all_symbols
        :rtype: list of sympy.Symbol
        """
        all_symbols = []

        for order in range(self.order):
            all_symbols.append(sympy.Symbol(str(self.symbol) + derivative_symbol * order))

        return all_symbols


    def get_all_variable_symbols(self, shapes=None, derivative_symbol="'"):
        r"""
        Get all variable symbols for this shape and all other shapes in :python:`shapes`, without duplicates, in no particular order.

        :return: all_symbols
        :rtype: list of sympy.Symbol
        """
        all_symbols = []
        all_shapes = []

        if not shapes is None:
            all_shapes = shapes

        if not self in all_shapes:
            all_shapes += [self]

        for shape in all_shapes:
            all_symbols.extend(shape.get_state_variables(derivative_symbol=derivative_symbol))

        all_symbols = list(set(all_symbols))		# filter for unique symbols

        return all_symbols


    def is_lin_const_coeff(self, shapes=None):
        r"""
        :return: :python:`True` if and only if the shape is linear and constant coefficient in all known variable symbols in :python:`shapes`.
        :rtype: bool
        """

        all_symbols = self.get_all_variable_symbols(shapes, derivative_symbol="__d")

        for sym in all_symbols:
            for df in self.derivative_factors:
                expr = sympy.diff(df, sym)
                if not sympy.sympify(expr).is_zero:
                    # the expression "sym * self.symbol" appears on right-hand side of this shape's definition
                    return False

            expr = sympy.diff(self.diff_rhs_derivatives, sym)
            if not sympy.sympify(expr).is_zero:
                # the variable symbol `sym` appears on right-hand side of this expression. Check to see if it appears as a linear term by checking whether taking its derivative again, with respect to any known variable, yields 0
                for sym_ in all_symbols:
                    if not sympy.sympify(sympy.diff(expr, sym_)).is_zero:
                        return False

        return True


    @classmethod
    def from_json(cls, indict, all_variable_symbols=None, time_symbol=sympy.Symbol("t"), differential_order_symbol="__d", _debug=False):
        r"""
        Create a :python:`Shape` instance from an input dictionary.

        :param indict: Input dictionary, i.e. one element of the :python:`"dynamics"` list supplied in the ODE-toolbox input dictionary.
        :param all_variable_symbols: All known variable symbols. :python:`None` or list of string.
        :param time_symbol: sympy Symbol representing the independent time variable.
        :param differential_order_symbol: String used for identifying differential order. XXX: only :python:`"__d"` is supported for now.
        """
        if not "expression" in indict:
            raise MalformedInputException("No `expression` keyword found in input")

        if not indict["expression"].count("=") == 1:
            raise MalformedInputException("Expecting exactly one \"=\" symbol in defining expression: \"" + str(indict["expression"]) + "\"")

        lhs, rhs = indict["expression"].split("=")
        lhs_ = re.findall(r"\S+", lhs)
        if not len(lhs_) == 1:
            raise MalformedInputException("Error while parsing expression \"" + indict["expression"] + "\"")
        lhs = lhs_[0]
        rhs = rhs.strip()

        symbol_match = re.search("[a-zA-Z_][a-zA-Z0-9_]*", lhs)
        if symbol_match is None:
            raise MalformedInputException("Error while parsing symbol name in \"" + lhs + "\"")
        symbol = symbol_match.group()
        order = len(re.findall("'", lhs))

        initial_values = {}
        if not "initial_value" in indict.keys() \
           and not "initial_values" in indict.keys() \
           and order > 0:
            raise MalformedInputException("No initial values specified for order " + str(order) + " equation with variable symbol \"" + symbol + "\"")

        if "initial_value" in indict.keys() \
           and "initial_values" in indict.keys():
            raise MalformedInputException("`initial_value` and `initial_values` cannot be specified simultaneously for equation with variable symbol \"" + symbol + "\"")

        if "initial_value" in indict.keys():
            if not order == 1:
                raise MalformedInputException("Single initial value specified for equation that is not first order in equation with variable symbol \"" + symbol + "\"")
            initial_values[symbol] = indict["initial_value"]

        if "initial_values" in indict.keys():
            if not len(indict["initial_values"]) == order:
                raise MalformedInputException("Wrong number of initial values specified for order " + str(order) + " equation with variable symbol \"" + symbol + "\"")

            initial_val_specified = [False] * order
            for iv_lhs, iv_rhs in indict["initial_values"].items():
                symbol_match = re.search("[a-zA-Z_][a-zA-Z0-9_]*", iv_lhs)
                if symbol_match is None:
                    raise MalformedInputException("Error trying to parse initial value variable symbol from string \"" + iv_lhs + "\"")
                iv_symbol = symbol_match.group()
                if not iv_symbol == symbol:
                    raise MalformedInputException("Initial value variable symbol \"" + iv_symbol + "\" does not match equation variable symbol \"" + symbol + "\"")
                iv_order = len(re.findall("'", iv_lhs))
                if iv_order >= order:
                    raise MalformedInputException("In defintion of initial value for variable \"" + iv_symbol + "\": differential order (" + str(iv_order) + ") exceeds that of overall equation order (" + str(order) + ")")
                if initial_val_specified[iv_order]:
                    raise MalformedInputException("Initial value for order " + str(iv_order) + " specified more than once")
                initial_val_specified[iv_order] = True
                initial_values[iv_symbol + iv_order * "'"] = iv_rhs

            if not all(initial_val_specified):
                raise MalformedInputException("Initial value not specified for all differential orders for variable \"" + iv_symbol + "\"")

        lower_bound = None
        if "lower_bound" in indict.keys():
            lower_bound = indict["lower_bound"]

        upper_bound = None
        if "upper_bound" in indict.keys():
            upper_bound = indict["upper_bound"]

        if order == 0:
            return Shape.from_function(symbol, rhs, differential_order_symbol=differential_order_symbol)
        else:
            return Shape.from_ode(symbol, rhs, initial_values, all_variable_symbols=all_variable_symbols, lower_bound=lower_bound, upper_bound=upper_bound, differential_order_symbol=differential_order_symbol)


    def reconstitute_expr(self, derivative_symbol="__d"):
        r"""
        Recreate SymPy expression from internal representation (linear coefficients and nonlinear part).

        :param differential_order_symbol: String used for identifying differential order. XXX: only :python:`"__d"` is supported for now.
        """
        expr = self.diff_rhs_derivatives
        derivative_symbols = self.get_state_variables(derivative_symbol=derivative_symbol)
        for derivative_factor, derivative_symbol in zip(self.derivative_factors, derivative_symbols):
            expr += derivative_factor * derivative_symbol
        logging.info("Shape " + str(self.symbol) + ": reconstituting expression " + str(expr))
        return expr


    @staticmethod
    def split_lin_nonlin(expr, x):
        r"""
        Split an expression into the form :python:`a_0 * x[0] + a_1 * x[1] + ... + c`. The coefficients :python:`a_0` ... :python:`a_n` are returned as :python:`lin_factors`. The nonlinear remainder is returned as :python:`nonlin_term`.
        """

        assert all([_is_sympy_type(sym) for sym in x])

        lin_factors = []
        logging.debug("Splitting expression " + str(expr) + " into symbols " + str(x))

        expr = expr.expand()
        for j, sym in enumerate(x):
            # check if there is a linear part in `sym`
            if expr.is_Add:
                terms = expr.args
            else:
                terms = [expr]

            # a term is linear in `sym` if `term/sym` contains only free symbols that are not in all_known_symbols, i.e. if the sets are disjoint
            linear_terms = [term for term in terms if (term / sym).free_symbols.isdisjoint(x)]
            if linear_terms:
                linear_factors = [term / sym for term in linear_terms]
                linear_factor = functools.reduce(lambda x, y: x + y, linear_factors)
                linear_terms = [term for term in linear_terms]
                linear_term = functools.reduce(lambda x, y: x + y, linear_terms)
            else:
                linear_factor = sympy.Float(0)
                linear_term = sympy.Float(0)

            lin_factors.append(linear_factor)
            expr = expr - linear_term

        lin_factors = np.array(lin_factors)
        nonlin_term = expr
        assert len(lin_factors) == len(x)

        logging.debug("\tlinear factors: " + str(lin_factors))
        logging.debug("\tnonlinear term: " + str(nonlin_term))

        return lin_factors, nonlin_term


    @classmethod
    def from_function(cls, symbol: str, definition, max_t=100, max_order=4, all_variable_symbols=None, time_symbol=sympy.Symbol("t"), differential_order_symbol=sympy.Symbol("__d"), debug=False):
        r"""
        Create a Shape object given a function of time.

        For a complete description of the algorithm, see https://ode-toolbox.readthedocs.io/en/latest/index.html#converting-direct-functions-of-time

        :param symbol: The variable name of the shape (e.g. :python:`"alpha"`, :python:`"I"`)
        :param definition: The definition of the shape (e.g. :python:`"(e/tau_syn_in) * t *  exp(-t/tau_syn_in)"`)
        :param all_variable_symbols: All known variable symbols. :python:`None` or list of string.

        :return: The canonical representation of the postsynaptic shape
        :rtype: Shape

        :Example:

        >>> Shape.from_function("I_in", "(e/tau) * t * exp(-t/tau)")
        """

        if all_variable_symbols is None:
            all_variable_symbols = []

        all_variable_symbols_dict = {str(el): el for el in all_variable_symbols}

        symbol = sympy.Symbol(symbol)
        definition = sympy.parsing.sympy_parser.parse_expr(definition, global_dict=Shape._sympy_globals, local_dict=all_variable_symbols_dict)

        # `derivatives` is a list of all derivatives of `shape` up to the order we are checking, starting at 0.
        derivatives = [definition, sympy.diff(definition, time_symbol)]

        logging.info("\nProcessing shape " + str(symbol) + " with defining expression = \"" + str(definition) + "\"")


        #
        #   to avoid a division by zero below, we have to find a `t` so that the shape function is not zero at this `t`.
        #

        t_val = None
        for t_ in range(0, max_t):
            if not definition.subs(time_symbol, t_).is_zero:
                t_val = t_
                break

        logging.debug("Found t: " + str(t_val))

        if t_val is None:

            #
            # it is very unlikely that the shape obeys a linear homogeneous ODE of order 1 and we still did not find a suitable `t_val`. This would mean that the function evaluates to zero at `t_` = 1, ..., `max_t`, which usually hints at an error in the specification of the function.
            #

            msg = "Cannot find t for which shape function is unequal to zero"
            raise Exception(msg)


        #
        #   first handle the case for an ODE of order 1, i.e. of the form I' = a0 * I
        #

        order = 1

        logging.debug("\tFinding ode for order 1...")

        derivative_factors = [(1 / derivatives[0] * derivatives[1]).subs(time_symbol, t_val)]
        diff_rhs_lhs = derivatives[1] - derivative_factors[0] * derivatives[0]
        found_ode = sympy.simplify(diff_rhs_lhs).is_zero


        #
        #   If `shape` does not satisfy a linear homogeneous ODE of order 1, we try to find one of higher order in a loop. The loop runs while no linear homogeneous ODE was found and the maximum order to check for was not yet reached.
        #

        while not found_ode and order < max_order:
            order += 1

            logging.debug("\tFinding ode for order " + str(order) + "...")

            # Add the next higher derivative to the list
            derivatives.append(sympy.diff(derivatives[-1], time_symbol))

            X = sympy.zeros(order)

            # `Y` is a vector of length `order` that will be assigned the derivatives of `order` of the natural number in the corresponding row of `X`
            Y = sympy.zeros(order, 1)

            # It is possible that by choosing certain natural numbers, the system of equations will not be solvable, i.e. `X` is not invertible. This is unlikely but we check for invertibility of `X` for varying sets of natural numbers.
            invertible = False
            for t_ in range(1, max_t):
                for i in range(order):
                    substitute = i + t_
                    Y[i] = derivatives[order].subs(time_symbol, substitute)
                    for j in range(order):
                        X[i, j] = derivatives[j].subs(time_symbol, substitute)

                if not sympy.simplify(sympy.det(X)).is_zero:
                    invertible = True
                    break

            #
            #   If we failed to find an invertible `X`, it is very unlikely that the shape function obeys a linear homogeneous ODE of order `order` and we go on checking the next potential order.
            #

            if not invertible:
                continue


            #
            #   calculate `derivative_factors`
            #

            derivative_factors = sympy.simplify(X.inv() * Y)


            #
            #   fill in the obtained expressions for the derivative_factors and check whether they satisfy the definition of the shape
            #

            diff_rhs_lhs = 0
            logging.debug("\tchecking whether shape definition is satisfied...")
            for k in range(order):
                diff_rhs_lhs -= derivative_factors[k] * derivatives[k]
            diff_rhs_lhs += derivatives[order]

            if len(str(diff_rhs_lhs)) < Shape.EXPRESSION_SIMPLIFICATION_THRESHOLD and sympy.simplify(diff_rhs_lhs).is_zero:
                found_ode = True
                break

        if not found_ode:
            raise Exception("Shape does not satisfy any ODE of order <= " + str(max_order))

        derivative_factors = [sympy.simplify(df) for df in derivative_factors]


        #
        #    calculate the initial values of the found ODE
        #

        initial_values = {str(symbol) + derivative_order * '\'': x.subs(time_symbol, 0) for derivative_order, x in enumerate(derivatives[:-1])}

        return cls(symbol, order, initial_values, derivative_factors)


    @classmethod
    def from_ode(cls, symbol: str, definition: str, initial_values: dict, all_variable_symbols=None, lower_bound=None, upper_bound=None, differential_order_symbol="__d", debug=False, **kwargs):
        r"""
        Create a :python:`Shape` object given an ODE and initial values.

        Note that shapes are only aware of their own state variables: if an equation for :math:`x` depends on another state variable :math:`y` of another shape, then :math:`y` will appear in the nonlinear part of :math:`x`.

        :param symbol: The symbol (variable name) of the ODE.
        :param definition: The definition of the ODE.
        :param initial_values: A dictionary mapping initial values to expressions.
        :param all_variable_symbols: All known variable symbols. :python:`None` or list of string.

        :Example:

        >>> Shape.from_ode("alpha",
                          "-1/tau**2 * shape_alpha -2/tau * shape_alpha'",
                          {"alpha" : "0", "alpha'" : "e/tau", "0"]})
        """

        assert type(symbol) is str
        assert type(definition) is str
        assert type(initial_values) is dict

        logging.info("\nProcessing shape " + str(symbol) + " with defining expression = \"" + str(definition) + "\"")

        if all_variable_symbols is None:
            all_variable_symbols = []

        order = len(initial_values)
        all_variable_symbols_dict = {str(el): el for el in all_variable_symbols}
        symbol = sympy.Symbol(symbol)
        definition = sympy.parsing.sympy_parser.parse_expr(definition.replace("'", differential_order_symbol), global_dict=Shape._sympy_globals, local_dict=all_variable_symbols_dict)  # minimal global_dict to make no assumptions (e.g. "beta" could otherwise be recognised as a function instead of as a parameter symbol)
        initial_values = {k: sympy.parsing.sympy_parser.parse_expr(v, global_dict=Shape._sympy_globals, local_dict=all_variable_symbols_dict) for k, v in initial_values.items()}

        local_symbols = [sympy.Symbol(str(symbol) + differential_order_symbol * i) for i in range(order)]
        if not symbol in all_variable_symbols:
            all_variable_symbols.extend(local_symbols)
        all_variable_symbols = [sympy.Symbol(str(sym_name).replace("'", differential_order_symbol)) for sym_name in all_variable_symbols]
        derivative_factors, diff_rhs_derivatives = Shape.split_lin_nonlin(definition, all_variable_symbols)
        local_symbols_idx = [all_variable_symbols.index(sym) for sym in local_symbols]
        local_derivative_factors = [derivative_factors[i] for i in local_symbols_idx]
        nonlocal_derivative_terms = [derivative_factors[i] * all_variable_symbols[i] for i in range(len(all_variable_symbols)) if i not in local_symbols_idx]
        if nonlocal_derivative_terms:
            diff_rhs_derivatives = diff_rhs_derivatives + functools.reduce(lambda x, y: x + y, nonlocal_derivative_terms)

        shape = cls(symbol, order, initial_values, local_derivative_factors, diff_rhs_derivatives, lower_bound, upper_bound)

        return shape
