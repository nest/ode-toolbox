#
# sympy_helpers.py
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

from typing import Mapping

import logging
import sympy
import sys

from .config import Config


class NumericalIssueException(Exception):
    r"""Thrown in case of numerical issues, like division by zero."""
    pass


def _is_constant_term(term, parameters: Mapping[sympy.Symbol, str] = None) -> bool:
    r"""
    :return: :python:`True` if and only if this term contains only numerical values and parameters; :python:`False` otherwise.
    """
    if parameters is None:
        parameters = {}

    assert all([type(k) is sympy.Symbol for k in parameters.keys()])

    if sympy.__version__.startswith("1.4"):
        sympy_zero = sympy.numbers.Zero
        sympy_one = sympy.numbers.One
    else:
        sympy_zero = sympy.core.numbers.Zero
        sympy_one = sympy.core.numbers.One

    return type(term) in [sympy.Float, sympy.Integer, sympy_zero, sympy_one] \
        or all([sym in parameters.keys() for sym in term.free_symbols])


def _check_numerical_issue(var: str) -> None:
    forbidden_vars = ["zoo", "oo", "nan", "NaN"]
    stripped_var_name = str(var).strip("'")
    if stripped_var_name in forbidden_vars:
        raise NumericalIssueException("The variable \"" + stripped_var_name + "\" was found. This indicates a numerical problem while solving the system of ODEs. Please check the input for correctness (such as the presence of divisions by zero).")


def _check_forbidden_name(var: str) -> None:
    from .shapes import MalformedInputException

    stripped_var_name = str(var).strip("'")
    if stripped_var_name in Config().forbidden_names:
        raise MalformedInputException("Variable by name \"" + stripped_var_name + "\" not allowed; this is a reserved name.")


def _is_zero(x):
    r"""
    Check if a sympy expression is equal to zero.

    In the ideal case, we would like to use sympy.simplify() to do simplification of an expression before comparing it to zero. However, for expressions of moderate size (e.g. a few dozen terms involving exp() functions), it becomes unbearably slow. We therefore use this internal function, so that the simplification function can be easily switched over.

    Tests by expand_mul only, suitable for polynomials and rational functions.

    Ref.: https://github.com/sympy/sympy PR #13877 by @normalhuman et al. merged on Jan 27, 2018
    """
    return bool(sympy.expand_mul(x).is_zero)


def _is_sympy_type(var):
    # for sympy version <= 1.4.*
    try:
        return isinstance(var, tuple(sympy.core.all_classes))
    except:  # noqa
        pass

    # for sympy version >= 1.5
    try:
        return isinstance(var, sympy.Basic)
    except:  # noqa
        pass

    raise Exception("Unsupported sympy version used")


def _custom_simplify_expr(expr: str):
    """Custom expression simplification"""
    if isinstance(expr, sympy.matrices.MatrixBase):
        return expr.applyfunc(_custom_simplify_expr)

    try:
        # skip simplification for long expressions
        if len(str(expr)) > Config().expression_simplification_threshold:
            logging.warning("Length of expression \"" + str(expr) + "\" exceeds sympy simplification threshold")

        _simplify_expr = compile(Config().simplify_expression, filename="<string>", mode="eval")
        expr_simplified = eval(_simplify_expr)

        return expr_simplified
    except Exception as e:
        print("Exception occurred while applying expression simplification function: " + type(e).__name__)
        print(str(e))
        print("Check that the parameter ``simplify_expression`` is properly formatted.")
        sys.exit(1)


def _find_in_matrix(A, el):
    num_rows = A.rows
    num_cols = A.cols

    # Iterate over the elements of the matrix
    for i in range(num_rows):
        for j in range(num_cols):
            if A[i, j] == el:
                return (i, j)

    return None


class SympyPrinter(sympy.printing.StrPrinter):

    def _print_Exp1(self, expr):
        return 'e'

    def _print_Function(self, expr):
        """
        Overrides base class method to print min() and max() functions in lowercase.
        """
        if expr.func.__name__ in ["Min", "Max"]:
            return expr.func.__name__.lower() + "(%s)" % self.stringify(expr.args, ", ")

        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")
