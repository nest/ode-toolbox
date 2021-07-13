#
# sympy_printer.py
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

import sympy
from sympy.printing import StrPrinter


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


class SympyPrinter(StrPrinter):

    def _print_Exp1(self, expr):
        return 'e'

    def _print_Function(self, expr):
        """
        Overrides base class method to print min() and max() functions in lowercase.
        """
        if expr.func.__name__ in ["Min", "Max"]:
            return expr.func.__name__.lower() + "(%s)" % self.stringify(expr.args, ", ")

        return expr.func.__name__ + "(%s)" % self.stringify(expr.args, ", ")
