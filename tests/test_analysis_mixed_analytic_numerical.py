#
# test_analysis_mixed_analytic_numerical.py
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
import pytest
import sympy
import sympy.parsing.sympy_parser


from .context import odetoolbox

try:
    import pygsl
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False


def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict



class TestAnalysisMixedAnalyticNumerical(unittest.TestCase):

    def test_mixed_analytic_numerical_no_stiffness(self):
        indict = open_json("mixed_analytic_numerical_no_stiffness.json")
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True)
        assert len(solver_dict) == 2
        assert (solver_dict[0]["solver"] == "analytical" and solver_dict[1]["solver"][:7] == "numeric") \
         or (solver_dict[1]["solver"] == "analytical" and solver_dict[0]["solver"][:7] == "numeric")

    @pytest.mark.skipif(not PYGSL_AVAILABLE, reason="Cannot run stiffness test if GSL is not installed.")
    def test_mixed_analytic_numerical_with_stiffness(self):
        indict = open_json("mixed_analytic_numerical_with_stiffness.json")
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=False)
        assert len(solver_dict) == 2
        assert (solver_dict[0]["solver"] == "analytical" and solver_dict[1]["solver"][:7] == "numeric") \
         or (solver_dict[1]["solver"] == "analytical" and solver_dict[0]["solver"][:7] == "numeric")


if __name__ == '__main__':
    unittest.main()
