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

from __future__ import annotations

from typing import List, Tuple

import functools
import logging
import re
import sympy
import sympy.parsing.sympy_parser

from sympy.core.expr import Expr as SympyExpr

from .config import Config
from .sympy_helpers import _check_numerical_issue, _check_forbidden_name, _custom_simplify_expr, _is_constant_term, _is_sympy_type, _is_zero


class MalformedInputException(Exception):
    """
    Thrown in case an error occurred while processing malformed input.
    """
    pass


class Shape:
    r"""
    This class provides a canonical representation of a shape function independently of the way in which the user specified the shape. It assumes a differential equation of the general form (where bracketed superscript :math:`{}^{(n)}` indicates the :math:`n`-th derivative with respect to time):

    .. math::

       x^{(n)} = N + \sum_{i=0}^{n-1} c_i x^{(i)}

    Any constant or nonlinear part is here contained in the term N.

    In the input and output, derivatives are indicated by adding one prime (single quotation mark) for each derivative order. For example, in the expression

    ::

       x''' = c0 + c1*x + c2*x' + c3*x'' + x*y + x**2

    the :python:`symbol` of the ODE would be :python:`x` (i.e. without any qualifiers), :python:`order` would be 3, :python:`derivative_factors` would contain the linear part in the form of the list :python:`[c1, c2, c3]`, the inhomogeneous term would be :python:`c0` and the nonlinear part :python:`x*y + x**2` is stored in :python:`nonlin_term`.
    """

    # a minimal subset of sympy classes and functions to avoid "from sympy import *"
    _sympy_globals = {"Symbol": sympy.Symbol,
                      "Integer": sympy.Integer,
                      "Float": sympy.Float,
                      "Function": sympy.Function,
                      "Add": sympy.Add,
                      "Mul": sympy.Mul,
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
                      "min": sympy.Min,
                      "max": sympy.Max,
                      "Heaviside": sympy.Heaviside,
                      "e": sympy.exp(1),
                      "E": sympy.exp(1),
                      "t": sympy.Symbol("t"),
                      "DiracDelta": sympy.DiracDelta}

    # cython backend (used by sympy autowrap()) cannot handle these functions; need to provide alternative implementation
    _sympy_autowrap_helpers = [("Min", (abs(sympy.symbols("x") + sympy.symbols("y")) - abs(sympy.symbols("x") - sympy.symbols("y"))) / 2, [sympy.symbols("x"), sympy.symbols("y")]),
                               ("Max", (abs(sympy.symbols("x") + sympy.symbols("y")) + abs(sympy.symbols("x") - sympy.symbols("y"))) / 2, [sympy.symbols("x"), sympy.symbols("y")]),
                               ("Heaviside", (sympy.symbols("x") + abs(sympy.symbols("x"))) / (2 * abs(sympy.symbols("x")) + 1E-300), [sympy.symbols("x")])]

    def __init__(self, symbol, order, initial_values, derivative_factors, inhom_term=sympy.Float(0.), nonlin_term=sympy.Float(0.), lower_bound=None, upper_bound=None):
        r"""
        Perform type and consistency checks and assign arguments to member variables.

        :param symbol: Symbolic name of the shape without additional qualifiers like prime symbols or similar.
        :param order: Order of the ODE representing the shape.
        :param initial_values: Initial values of the ODE representing the shape. The dict contains :python:`order` many key-value pairs: one for each derivative that occurs in the ODE. The keys are strings created by concatenating the variable symbol with as many single quotation marks (') as the derivation order. The values are SymPy expressions.
        :param derivative_factors: The factors for the derivatives that occur in the ODE. This list has to contain :python:`order` many values, i.e. one for each derivative that occurs in the ODE. The values have to be in ascending order, i.e. :python:`[c1, c2, c3]` for the given example.
        :param inhom_term: Inhomogeneous part of the ODE representing the shape, i.e. :python:`c0` for the given example.
        :param nonlin_term: Nonlinear part of the ODE representing the shape, i.e. :python:`x*y + x**2` for the given example.
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
            if Config().differential_order_symbol in str(sym):
                raise MalformedInputException("Input variable name \"" + str(sym) + "\" should not contain the string \"" + Config().differential_order_symbol + "\", which is used to indicate variable differential order")

        if not len(derivative_factors) == order:
            raise MalformedInputException(str(len(derivative_factors)) + " derivative factors specified, while " + str(order) + " were expected based on differential equation definition")

        for df in derivative_factors:
            if not _is_sympy_type(df):
                raise MalformedInputException("Derivative factor \"%r\" is not a SymPy expression" % df)

        self.derivative_factors = derivative_factors
        self.inhom_term = _custom_simplify_expr(inhom_term)
        self.nonlin_term = _custom_simplify_expr(nonlin_term)

        self.lower_bound = lower_bound
        if not self.lower_bound is None:
            self.lower_bound = _custom_simplify_expr(self.lower_bound)

        self.upper_bound = upper_bound
        if not self.upper_bound is None:
            self.upper_bound = _custom_simplify_expr(self.upper_bound)

        logging.debug("Created Shape with symbol " + str(self.symbol) + ", derivative_factors = " + str(self.derivative_factors) + ", inhom_term = " + str(self.inhom_term) + ", nonlin_term = " + str(self.nonlin_term))


    def __str__(self):
        s = "Shape \"" + str(self.symbol) + "\" of order " + str(self.order)
        return s


    def is_homogeneous(self) -> bool:
        r"""
        :return: :python:`False` if and only if the shape has a nonzero right-hand side.
        """
        return _is_zero(self.inhom_term)


    def get_initial_value(self, sym: str):
        r"""
        Get the initial value corresponding to the variable symbol.

        :param sym: String representation of a sympy symbol, e.g. :python:`"V_m'"`
        """
        if not sym in self.initial_values.keys():
            return None
        return self.initial_values[sym]


    def get_state_variables(self, derivative_symbol="'") -> List[sympy.Symbol]:
        r"""
        Get all variable symbols for this shape, ordered according to derivative order, up to the shape's order :math:`N`: :python:`[sym, dsym/dt, d^2sym/dt^2, ..., d^(N-1)sym/dt^(N-1)]`
        """
        all_symbols = []

        for order in range(self.order):
            all_symbols.append(sympy.Symbol(str(self.symbol) + derivative_symbol * order))

        return all_symbols


    def get_all_variable_symbols(self, shapes=None, derivative_symbol="'") -> List[sympy.Symbol]:
        r"""
        Get all variable symbols for this shape and all other shapes in :python:`shapes`, without duplicates, in no particular order.

        :return: all_symbols
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


    def is_lin_const_coeff(self) -> bool:
        r"""
        :return: :python:`True` if and only if the shape is linear and constant coefficient.
        """
        return _is_zero(self.nonlin_term)


    def is_lin_const_coeff_in(self, symbols, parameters=None):
        r"""
        :return: :python:`True` if and only if the shape is linear and constant coefficient in those variables passed in ``symbols``.
        """
        expr = self.reconstitute_expr()
        derivative_factors, inhom_term, nonlin_term = Shape.split_lin_inhom_nonlin(expr, symbols, parameters=parameters)
        return _is_zero(nonlin_term)


    @classmethod
    def _parse_defining_expression(cls, s: str) -> Tuple[str, int, str]:
        r"""Parse a defining expression, for example, if the ODE-toolbox JSON input file contains the snippet:

        ::
           {
               "expression": "h' = -g / tau**2 - 2 * h / tau"
           }

        then the corresponding defining expression is ``"h' = -g / tau**2 - 2 * h / tau"``.

        This function parses that string and returns the variable name (``h``), the derivative order (1) and the right-hand side expression (``-g / tau**2 - 2 * h / tau"``).
        """
        lhs, rhs = s.split("=")
        lhs_ = re.findall(r"\S+", lhs)
        if not len(lhs_) == 1:
            raise MalformedInputException("Error while parsing expression \"" + s + "\"")
        lhs = lhs_[0]
        rhs = rhs.strip()

        symbol_match = re.search("[a-zA-Z_][a-zA-Z0-9_]*", s)
        if symbol_match is None:
            raise MalformedInputException("Error while parsing symbol name in \"" + lhs + "\"")
        symbol = symbol_match.group()

        order = len(re.findall("'", lhs))
        return symbol, order, rhs


    @classmethod
    def from_json(cls, indict, all_variable_symbols=None, parameters=None, _debug=False):
        r"""
        Create a :python:`Shape` instance from an input dictionary.

        :param indict: Input dictionary, i.e. one element of the :python:`"dynamics"` list supplied in the ODE-toolbox input dictionary.
        :param all_variable_symbols: All known variable symbols. :python:`None` or list of string.
        :param parameters: An optional dictionary of parameters to their defining expressions.
        """
        if not "expression" in indict:
            raise MalformedInputException("No `expression` keyword found in input")

        if not indict["expression"].count("=") == 1:
            raise MalformedInputException("Expecting exactly one \"=\" symbol in defining expression: \"" + str(indict["expression"]) + "\"")

        symbol, order, rhs = Shape._parse_defining_expression(indict["expression"])

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

                _check_forbidden_name(iv_symbol)

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
            return Shape.from_function(symbol, rhs)

        return Shape.from_ode(symbol, rhs, initial_values, all_variable_symbols=all_variable_symbols, lower_bound=lower_bound, upper_bound=upper_bound, parameters=parameters)


    def reconstitute_expr(self) -> SympyExpr:
        r"""
        Recreate right-hand side expression from internal representation (linear coefficients, inhomogeneous, and nonlinear parts).
        """
        expr = self.inhom_term + self.nonlin_term
        derivative_symbols = self.get_state_variables(derivative_symbol=Config().differential_order_symbol)
        for derivative_factor, derivative_symbol in zip(self.derivative_factors, derivative_symbols):
            expr += derivative_factor * derivative_symbol
        logging.info("Shape " + str(self.symbol) + ": reconstituting expression " + str(expr))
        return expr


    @staticmethod
    def split_lin_inhom_nonlin(expr, x, parameters=None):
        r"""
        Split an expression into a linear, inhomogeneous and nonlinear part.

        For example, in the expression

        ::

           x''' = c0 + c1*x + c2*x' + c3*x'' + x*y + x**2

        :python:`lin_factors` would contain the linear part in the form of the list :python:`[c1, c2, c3]`, the inhomogeneous term would be :python:`c0` and the nonlinear part :python:`x*y + x**2` is returned as :python:`nonlin_term`.

        All parameters in :python:`parameters` are assumed to be constants.
        """

        assert all([_is_sympy_type(sym) for sym in x])

        if parameters is None:
            parameters = {}

        logging.debug("Splitting expression " + str(expr) + " (symbols " + str(x) + ")")

        lin_factors = sympy.zeros(len(x), 1)
        inhom_term = sympy.Float(0)
        nonlin_term = sympy.Float(0)

        expr = expr.expand()
        if expr.is_Add:
            terms = expr.args
        else:
            terms = [expr]

        for term in terms:
            if _is_constant_term(term, parameters=parameters):
                inhom_term += term
            else:
                # check if the term is linear in any of the symbols in `x`
                is_lin = False
                for j, sym in enumerate(x):
                    if _is_constant_term(term / sym, parameters=parameters):
                        lin_factors[j] += term / sym
                        is_lin = True
                        break
                if not is_lin:
                    nonlin_term += term

        logging.debug("\tlinear factors: " + str(lin_factors))
        logging.debug("\tinhomogeneous term: " + str(inhom_term))
        logging.debug("\tnonlinear term: " + str(nonlin_term))

        return lin_factors, inhom_term, nonlin_term


    @classmethod
    def from_function(cls, symbol: str, definition, max_t=100, max_order=4, all_variable_symbols=None, debug=False) -> Shape:
        r"""
        Create a Shape object given a function of time.

        Only functions of time that have a homogeneous ODE equivalent are supported (inhomogeneous ODE functions are not supported).

        For a complete description of the algorithm, see https://ode-toolbox.readthedocs.io/en/latest/index.html#converting-direct-functions-of-time Note that this function uses sympy.simplify() rather than the custom simplification expression, because the latter risks that we fail to recognize certain shapes.

        :param symbol: The variable name of the shape (e.g. :python:`"alpha"`, :python:`"I"`)
        :param definition: The definition of the shape (e.g. :python:`"(e/tau_syn_in) * t *  exp(-t/tau_syn_in)"`)
        :param all_variable_symbols: All known variable symbols. :python:`None` or list of string.

        :return: The canonical representation of the postsynaptic shape

        :Example:

        >>> Shape.from_function("I_in", "(e/tau) * t * exp(-t/tau)")
        """

        if all_variable_symbols is None:
            all_variable_symbols = []

        all_variable_symbols_dict = {str(el): el for el in all_variable_symbols}

        definition = sympy.parsing.sympy_parser.parse_expr(definition, global_dict=Shape._sympy_globals, local_dict=all_variable_symbols_dict)

        # `derivatives` is a list of all derivatives of `shape` up to the order we are checking, starting at 0.
        derivatives = [definition, sympy.diff(definition, Config().input_time_symbol)]

        logging.info("\nProcessing function-of-time shape \"" + symbol + "\" with defining expression = \"" + str(definition) + "\"")


        #
        #   to avoid a division by zero below, we have to find a `t` so that the shape function is not zero at this `t`.
        #

        t_val = None
        for t_ in range(0, max_t):
            if not _is_zero(definition.subs(Config().input_time_symbol, t_)):
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

        derivative_factors = [(1 / derivatives[0] * derivatives[1]).subs(Config().input_time_symbol, t_val)]
        diff_rhs_lhs = derivatives[1] - derivative_factors[0] * derivatives[0]
        found_ode = _is_zero(diff_rhs_lhs)


        #
        #   If `shape` does not satisfy a linear homogeneous ODE of order 1, we try to find one of higher order in a loop. The loop runs while no linear homogeneous ODE was found and the maximum order to check for was not yet reached.
        #

        while not found_ode and order < max_order:
            order += 1

            logging.debug("\tFinding ode for order " + str(order) + "...")

            # Add the next higher derivative to the list
            derivatives.append(sympy.diff(derivatives[-1], Config().input_time_symbol))

            X = sympy.zeros(order)

            # `Y` is a vector of length `order` that will be assigned the derivatives of `order` of the natural number in the corresponding row of `X`
            Y = sympy.zeros(order, 1)

            # It is possible that by choosing certain natural numbers, the system of equations will not be solvable, i.e. `X` is not invertible. This is unlikely but we check for invertibility of `X` for varying sets of natural numbers.
            invertible = False
            for t_ in range(1, max_t):
                for i in range(order):
                    substitute = i + t_
                    Y[i] = derivatives[order].subs(Config().input_time_symbol, substitute)
                    for j in range(order):
                        X[i, j] = derivatives[j].subs(Config().input_time_symbol, substitute)

                if not _is_zero(sympy.det(X)):
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

            derivative_factors = sympy.simplify(X.inv() * Y)    # XXX: need sympy.simplify() here rather than _custom_simplify_expr()


            #
            #   fill in the obtained expressions for the derivative_factors and check whether they satisfy the definition of the shape
            #

            diff_rhs_lhs = 0
            logging.debug("\tchecking whether shape definition is satisfied...")
            for k in range(order):
                diff_rhs_lhs -= derivative_factors[k] * derivatives[k]
            diff_rhs_lhs += derivatives[order]

            if _is_zero(sympy.simplify(diff_rhs_lhs)):    # XXX: need sympy.simplify() here rather than _custom_simplify_expr()
                found_ode = True
                break

        if not found_ode:
            raise Exception("Shape does not satisfy any ODE of order <= " + str(max_order))

        logging.debug("Shape satisfies ODE of order = " + str(order))

        #
        #    calculate the initial values of the found ODE
        #

        initial_values = {symbol + derivative_order * "'": x.subs(Config().input_time_symbol, 0) for derivative_order, x in enumerate(derivatives[:-1])}

        return cls(sympy.Symbol(symbol), order, initial_values, derivative_factors)


    @classmethod
    def from_ode(cls, symbol: str, definition: str, initial_values: dict, all_variable_symbols=None, lower_bound=None, upper_bound=None, parameters=None, debug=False, **kwargs) -> Shape:
        r"""
        Create a :python:`Shape` object given an ODE and initial values.

        Note that shapes are only aware of their own state variables: if an equation for :math:`x` depends on another state variable :math:`y` of another shape, then :math:`y` will be assumed to be a parameter and the term will appear in the inhomogeneous component of :math:`x`.

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

        logging.info("\nProcessing differential-equation form shape " + str(symbol) + " with defining expression = \"" + str(definition) + "\"")

        if all_variable_symbols is None:
            all_variable_symbols = []

        order: int = len(initial_values)
        all_variable_symbols_dict = {str(el): el for el in all_variable_symbols}
        definition = sympy.parsing.sympy_parser.parse_expr(definition.replace("'", Config().differential_order_symbol), global_dict=Shape._sympy_globals, local_dict=all_variable_symbols_dict)  # minimal global_dict to make no assumptions (e.g. "beta" could otherwise be recognised as a function instead of as a parameter symbol)

        # validate input for forbidden names
        _initial_values = {k: sympy.parsing.sympy_parser.parse_expr(v, evaluate=False, global_dict=Shape._sympy_globals, local_dict=all_variable_symbols_dict) for k, v in initial_values.items()}
        for iv_expr in _initial_values.values():
            for var in iv_expr.atoms():
                _check_forbidden_name(var)

        # parse input
        initial_values = {k: sympy.parsing.sympy_parser.parse_expr(v, global_dict=Shape._sympy_globals, local_dict=all_variable_symbols_dict) for k, v in initial_values.items()}

        # validate input for numerical issues
        for iv_expr in initial_values.values():
            for var in iv_expr.atoms():
                _check_numerical_issue(var)

        local_symbols = [symbol + Config().differential_order_symbol * i for i in range(order)]
        local_symbols_sympy = [sympy.Symbol(sym_name) for sym_name in local_symbols]
        if not symbol in all_variable_symbols:
            all_variable_symbols.extend(local_symbols_sympy)
        all_variable_symbols = [str(sym_name).replace("'", Config().differential_order_symbol) for sym_name in all_variable_symbols]
        all_variable_symbols_sympy = [sympy.Symbol(sym_name) for sym_name in all_variable_symbols]
        derivative_factors, inhom_term, nonlin_term = Shape.split_lin_inhom_nonlin(definition, all_variable_symbols_sympy, parameters=parameters)
        local_symbols_idx = [all_variable_symbols.index(sym) for sym in local_symbols]
        local_derivative_factors = [derivative_factors[i] for i in local_symbols_idx]
        nonlocal_derivative_terms = [derivative_factors[i] * all_variable_symbols_sympy[i] for i in range(len(all_variable_symbols)) if i not in local_symbols_idx]
        if nonlocal_derivative_terms:
            nonlin_term = nonlin_term + functools.reduce(lambda x, y: x + y, nonlocal_derivative_terms)

        shape = cls(sympy.Symbol(symbol), order, initial_values, local_derivative_factors, inhom_term, nonlin_term, lower_bound, upper_bound)
        logging.info("\tReturning shape: " + str(shape))

        return shape
