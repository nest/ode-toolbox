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

import unittest

from .context import odetoolbox
from odetoolbox.shapes import Shape


class TestShapeFunction(unittest.TestCase):

    def test_shape_to_odes(self):
        shape_inh = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = Shape.from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        self.assertIsNotNone(shape_inh.ode_definition)
        self.assertIsNotNone(shape_exc.ode_definition)


class TestShapeODE(unittest.TestCase):

    def test_ode_shape(self):

        shape_inh = Shape.from_ode("alpha", "-1/tau**2 * alpha -2/tau * alpha'", ["0", "e/tau"])
        self.assertIsNotNone(shape_inh.derivative_factors)

        
if __name__ == '__main__':
    unittest.main()
