#
# test_sympy_helpers.py
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
import sympy

from odetoolbox.config import Config
from odetoolbox.sympy_helpers import _sympy_floats_to_ints


class TestSympyFloatsToInts:
    r"""Test the function ``_sympy_floats_to_ints()``."""

    x = sympy.Symbol("x", real=True)

    @pytest.mark.parametrize("s, expected", [("2.0", 2),
                                             ("-3.0", -3),
                                             ("0.0", 0),
                                             ("1E3", 1000),
                                             ("8388607.0", 2**23 - 1)])
    def test_integer_valued_floats_are_converted(self, s, expected):
        result = _sympy_floats_to_ints(sympy.Float(s))
        assert isinstance(result, sympy.Integer)
        assert result == expected

    @pytest.mark.parametrize("s", ["2.5",
                                   "-0.1",
                                   "1E-3",
                                   "2.9999999999999999999",    # rounds to 3.0 as a double; must not be truncated to 2
                                   "2.0000000000000000001"])   # rounds to 2.0 as a double, but is not an integer
    def test_non_integer_floats_are_untouched(self, s):
        f = sympy.Float(s, 30)
        result = _sympy_floats_to_ints(f)
        assert isinstance(result, sympy.Float)
        assert result == f

    @pytest.mark.parametrize("s", ["8388608.0",     # exactly 2**23
                                   "-8388608.0",
                                   "1E23",
                                   "2.5E30",
                                   "1E400"])        # exceeds the double-precision range
    def test_large_floats_are_untouched(self, s):
        f = sympy.Float(s)
        result = _sympy_floats_to_ints(f)
        assert isinstance(result, sympy.Float)
        assert result == f

    def test_magnitude_threshold_follows_config(self):
        f = sympy.Float(2**23)
        assert isinstance(_sympy_floats_to_ints(f), sympy.Float)

        # ``Config.config`` is global state shared between tests; always restore the original value
        orig_max_float_to_int_bits = Config.config["max_float_to_int_bits"]
        try:
            Config.config["max_float_to_int_bits"] = 53
            result = _sympy_floats_to_ints(f)
            assert isinstance(result, sympy.Integer)
            assert result == 2**23

            Config.config["max_float_to_int_bits"] = 1
            assert _sympy_floats_to_ints(sympy.Float(1.)) == sympy.Integer(1)
            assert isinstance(_sympy_floats_to_ints(sympy.Float(2.)), sympy.Float)
        finally:
            Config.config["max_float_to_int_bits"] = orig_max_float_to_int_bits

    def test_nested_expression(self):
        t = sympy.Symbol("t", real=True)
        expr = 2.0 * self.x**3.0 + 0.5 * sympy.exp(-1.0 * t)
        assert _sympy_floats_to_ints(expr) == 2 * self.x**3 + 0.5 * sympy.exp(-t)

    def test_matrix(self):
        A = sympy.Matrix([[-1.0 * self.x, 0.5],
                          [4.0, 1E30]])
        result = _sympy_floats_to_ints(A)
        assert result == sympy.Matrix([[-self.x, sympy.Float(0.5)],
                                       [sympy.Integer(4), sympy.Float(1E30)]])
        assert isinstance(result[1, 0], sympy.Integer)
        assert isinstance(result[1, 1], sympy.Float)

    def test_expression_without_floats_is_unchanged(self):
        expr = 2 * self.x**3 + sympy.Rational(1, 3)
        assert _sympy_floats_to_ints(expr) == expr
