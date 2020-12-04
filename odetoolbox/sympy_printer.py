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
