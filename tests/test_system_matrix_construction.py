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

import json
import os
import unittest
import sympy

from odetoolbox import _from_json_to_shapes, default_config
from odetoolbox.system_of_shapes import SystemOfShapes


def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestSystemMatrixConstruction(unittest.TestCase):

    def test_system_matrix_construction(self):
        indict = open_json("system_matrix_test.json")
        shapes = _from_json_to_shapes(indict, default_config)
        sigma, beta = sympy.symbols("sigma beta")
        parameters = {sympy.Symbol(k): v for k, v in indict["parameters"].items()}
        shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
        assert shape_sys.A_ == sympy.Matrix([[-sigma, sigma, 0.0],
                                             [0.0, 0.0, 0.0],
                                             [0.0, 0.0, -beta]])
        x, y, z = sympy.symbols("x y z")
        assert shape_sys.c_ == sympy.Matrix([[0],
                                             [3 * z * x**2 - x * y],
                                             [x * y]])


    def test_lorenz_attractor(self):
        indict = open_json("lorenz_attractor.json")
        shapes = _from_json_to_shapes(indict, default_config)
        sigma, beta, rho = sympy.symbols("sigma beta rho")
        parameters = {sympy.Symbol(k): v for k, v in indict["parameters"].items()}
        shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
        assert shape_sys.A_ == sympy.Matrix([[-sigma, sigma, 0],
                                             [rho, -1, 0],
                                             [0, 0, -beta]])
        x, y, z = sympy.symbols("x y z")
        assert shape_sys.c_ == sympy.Matrix([[0],
                                             [-x * z],
                                             [x * y]])


    def test_morris_lecar(self):
        indict = open_json("morris_lecar.json")
        shapes = _from_json_to_shapes(indict, default_config)
        parameters = {sympy.Symbol(k): v for k, v in indict["parameters"].items()}
        shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
        C_m, g_Ca, g_K, g_L, E_Ca, E_K, E_L, I_ext = sympy.symbols("C_m g_Ca g_K g_L E_Ca E_K E_L I_ext")
        assert shape_sys.A_ == sympy.Matrix([[-500.0 * g_Ca / C_m - 1000.0 * g_L / C_m, 1000.0 * E_K * g_K / C_m],
                                            [0.0, 0.0]])

        V, W = sympy.symbols("V W")
        assert shape_sys.b_ == sympy.Matrix([[500.0*E_Ca*g_Ca/C_m + 1000.0*E_L*g_L/C_m + 1000.0*I_ext/C_m],
                                             [0.0]])
        assert shape_sys.c_ == sympy.Matrix([[500.0*E_Ca*g_Ca*sympy.tanh(V/15 + 1/15)/C_m - 1000.0*V*W*g_K/C_m - 500.0*V*g_Ca*sympy.tanh(V/15 + 1/15)/C_m],
                                             [-200.0*W*sympy.cosh(V/60) + 100.0*sympy.cosh(V/60)*sympy.tanh(V/30) + 100.0*sympy.cosh(V/60)]])


if __name__ == '__main__':
    unittest.main()
