#
# test_lorenz_attractor.py
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
import sympy.parsing.sympy_parser

from tests.test_utils import _open_json

from .context import odetoolbox
from odetoolbox.shapes import Shape

try:
    import pygsl
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False


class TestLorenzAttractor:
    def test_lorenz_attractor(self):
        indict = _open_json("lorenz_attractor.json")
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=not PYGSL_AVAILABLE)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"].startswith("numeric")
        assert sympy.parsing.sympy_parser.parse_expr(solver_dict["update_expressions"]["x"], global_dict=Shape._sympy_globals).expand().simplify() \
               == sympy.parsing.sympy_parser.parse_expr("sigma*(-x + y)", global_dict=Shape._sympy_globals).expand().simplify()
        assert sympy.parsing.sympy_parser.parse_expr(solver_dict["update_expressions"]["y"], global_dict=Shape._sympy_globals).expand().simplify() \
               == sympy.parsing.sympy_parser.parse_expr("rho*x - x*z - y", global_dict=Shape._sympy_globals).expand().simplify()
        assert sympy.parsing.sympy_parser.parse_expr(solver_dict["update_expressions"]["z"], global_dict=Shape._sympy_globals).expand().simplify() \
               == sympy.parsing.sympy_parser.parse_expr("-beta*z + x*y", global_dict=Shape._sympy_globals).expand().simplify()
