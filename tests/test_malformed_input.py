#
# test_malformed_input.py
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

import pytest

from .context import odetoolbox
from odetoolbox.shapes import MalformedInputException
from odetoolbox.sympy_helpers import NumericalIssueException


class TestMalformedInput:
    r"""Test for failure when forbidden names are used in the input."""

    @pytest.mark.xfail(strict=True, raises=MalformedInputException)
    def test_malformed_input_iv(self):
        indict = {"dynamics": [{"expression": "x' = 0",
                                "initial_value": "zoo"}]}
        result = odetoolbox.analysis(indict, disable_stiffness_check=True)

    @pytest.mark.xfail(strict=True, raises=MalformedInputException)
    def test_malformed_input_expr(self):
        indict = {"dynamics": [{"expression": "x' = 42 * NaN",
                                "initial_value": "0."}]}
        result = odetoolbox.analysis(indict, disable_stiffness_check=True)

    @pytest.mark.xfail(strict=True, raises=MalformedInputException)
    def test_malformed_input_sym(self):
        indict = {"dynamics": [{"expression": "oo' = 0",
                                "initial_value": "0."}]}
        result = odetoolbox.analysis(indict, disable_stiffness_check=True)

    def test_correct_input(self):
        indict = {"dynamics": [{"expression": "foo' = 0",
                                "initial_value": "0."}]}
        result = odetoolbox.analysis(indict, disable_stiffness_check=True)

    @pytest.mark.xfail(strict=True, raises=NumericalIssueException)
    def test_malformed_input_numerical_iv(self):
        indict = {"dynamics": [{"expression": "foo' = 0",
                                "initial_value": "1/0"}]}
        result = odetoolbox.analysis(indict, disable_stiffness_check=True)

    @pytest.mark.xfail(strict=True, raises=NumericalIssueException)
    def test_malformed_input_numerical_parameter(self):
        indict = {"dynamics": [{"expression": "foo' = bar",
                                "initial_value": "1"}],
                  "parameters": {"bar": "1/0"}}
        result = odetoolbox.analysis(indict, disable_stiffness_check=True)
