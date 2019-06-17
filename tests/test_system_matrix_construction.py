#
# tes
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
import numpy as np
#np.seterr(under="warn")

from .context import odetoolbox
from odetoolbox import from_json_to_shapes, default_config
from odetoolbox.system_of_shapes import SystemOfShapes
from odetoolbox.analytic_integrator import AnalyticIntegrator

from math import e
from sympy import exp, sympify

import scipy
import scipy.special
import scipy.linalg
from scipy.integrate import solve_ivp


def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestSystemMatrixConstruction(unittest.TestCase):

    def test_system_matrix_construction(self):
        indict = open_json("system_matrix_test.json")
        _, _, shapes = from_json_to_shapes(indict, default_config)
        shape_sys = SystemOfShapes.from_shapes(shapes)
        sigma, beta = sympy.symbols("sigma beta")
        assert shape_sys.A_ == sympy.Matrix([[-sigma, sigma,     0], \
                                             [   0.0,   0.0,   0.0], \
                                             [   0.0,   0.0, -beta]])
        x, y, z = sympy.symbols("x y z")
        assert shape_sys.C_ == sympy.Matrix([[             0], \
                                             [3*z*x**2 - x*y], \
                                             [           x*y]])

    def test_lorenz_attractor(self):
        indict = open_json("lorenz_attractor.json")
        _, _, shapes = from_json_to_shapes(indict, default_config)
        shape_sys = SystemOfShapes.from_shapes(shapes)
        sigma, beta, rho = sympy.symbols("sigma beta rho")
        assert shape_sys.A_ == sympy.Matrix([[-sigma, sigma,     0], \
                                             [   rho,    -1,     0], \
                                             [     0,     0, -beta]])
        x, y, z = sympy.symbols("x y z")
        assert shape_sys.C_ == sympy.Matrix([[   0], \
                                             [-x*z], \
                                             [ x*y]])


if __name__ == '__main__':
    unittest.main()
