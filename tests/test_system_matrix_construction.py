#
# test_system_matrix_construction.py
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

from odetoolbox import _from_json_to_shapes
from odetoolbox.sympy_helpers import _sympy_floats_to_ints, _sympy_parse_real
from odetoolbox.system_of_shapes import SystemOfShapes
from tests.test_utils import load_test_json


class TestSystemMatrixConstruction:

    def test_system_matrix_construction(self):
        indict = load_test_json("system_matrix_test.json")
        shapes, parameters = _from_json_to_shapes(indict)
        sigma, beta = sympy.symbols("sigma beta", real=True)
        shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
        assert shape_sys.A_ == sympy.Matrix([[-sigma, sigma, 0],
                                             [0, 0, 0],
                                             [0, 0, -beta]])
        x, y, z = sympy.symbols("x y z", real=True)
        assert shape_sys.c_ == sympy.Matrix([[0],
                                             [3 * z * x**2 - x * y],
                                             [x * y]])

    def test_lorenz_attractor(self):
        indict = load_test_json("lorenz_attractor.json")
        shapes, parameters = _from_json_to_shapes(indict)
        sigma, beta, rho = sympy.symbols("sigma beta rho", real=True)
        shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
        assert shape_sys.A_ == sympy.Matrix([[-sigma, sigma, 0],
                                             [rho, -1, 0],
                                             [0, 0, -beta]])
        x, y, z = sympy.symbols("x y z", real=True)
        assert shape_sys.c_ == sympy.Matrix([[0],
                                             [-x * z],
                                             [x * y]])

    def test_morris_lecar(self):
        r"""The decimal input (``.5``, ``1E3``) yields a mix of integer-valued Floats and Integers in the system matrices (for instance, ``expand()`` turns ``0.5 * 1000`` into ``500.0``). Since sympy 1.13, ``Float(500.0) != Integer(500)``, so integer-valued Floats are converted to Integers before comparing."""
        indict = load_test_json("morris_lecar.json")
        shapes, parameters = _from_json_to_shapes(indict)
        shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
        shape_sys.A_ = _sympy_floats_to_ints(shape_sys.A_)
        shape_sys.b_ = _sympy_floats_to_ints(shape_sys.b_)
        shape_sys.c_ = _sympy_floats_to_ints(shape_sys.c_)

        C_m, g_Ca, g_K, g_L, E_Ca, E_K, E_L, I_ext = sympy.symbols("C_m g_Ca g_K g_L E_Ca E_K E_L I_ext", real=True)
        assert shape_sys.A_ == sympy.Matrix([[_sympy_parse_real("-500 * g_Ca / C_m - 1000 * g_L / C_m"), _sympy_parse_real("1000 * E_K * g_K / C_m")],
                                            [_sympy_parse_real("0"), _sympy_parse_real("0")]])

        V, W = sympy.symbols("V W", real=True)
        assert shape_sys.b_ == sympy.Matrix([[_sympy_parse_real("500 * E_Ca * g_Ca / C_m + 1000 * E_L * g_L / C_m + 1000 * I_ext / C_m")],
                                             [_sympy_parse_real("0")]])
        assert shape_sys.c_ == sympy.Matrix([[_sympy_parse_real("500 * E_Ca * g_Ca * tanh(V / 15 + 1 / 15) / C_m - 1000 * V * W * g_K / C_m - 500 * V * g_Ca * tanh(V / 15 + 1 / 15) / C_m")],
                                             [_sympy_parse_real("-200 * W * cosh(V / 60) + 100 * cosh(V / 60) * tanh(V / 30) + 100 * cosh(V / 60)")]])
