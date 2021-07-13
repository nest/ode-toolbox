#
# test_shapes.py
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

from odetoolbox.shapes import Shape


def test_ode_shape():
    shape_inh = Shape.from_ode("alpha", "-1/tau**2 * alpha -2/tau * alpha'", {"alpha": "0", "alpha'": "e/tau"})
    assert not shape_inh.derivative_factors is None


def test_ode_shape_fails_too_high_order_deriv():
    with pytest.raises(Exception):
        Shape.from_ode("alpha", "-1/tau**2 * alpha -2/tau * alpha'", {"alpha": "0", "alpha''": "e/tau"})


def test_ode_shape_fails_missing_deriv():
    with pytest.raises(Exception):
        Shape.from_ode("alpha", "-1/tau**2 * alpha -2/tau * alpha'", {"alpha'": "e/tau"})


def test_ode_shape_fails_unknown_symbol():
    with pytest.raises(Exception):
        Shape.from_ode("alpha", "-1/tau**2 * alpha -2/tau * alpha'", {"xyz": "0", "alpha": "e/tau"})
