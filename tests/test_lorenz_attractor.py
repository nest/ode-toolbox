#
# test_integration.py
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


class TestLorenzAttractor(unittest.TestCase):
    def test_lorenz_attractor(self):
        indict = open_json("lorenz_attractor.json")
        solver_dict = odetoolbox.analysis(indict)
        print("Got solver_dict from ode-toolbox: ")
        print(json.dumps(solver_dict,  indent=2))
        assert len(solver_dict) == 2
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"


if __name__ == '__main__':
    unittest.main()
