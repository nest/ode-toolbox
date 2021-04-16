#
# test_homogeneous.py
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

from odetoolbox.shapes import Shape


class TestHomogeneous(unittest.TestCase):

    def test_homogeneous(self):
        shape_inh = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = Shape.from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shape_V_m = Shape.from_ode("V_m", "-V_m/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."})

        self.assertTrue(shape_inh.is_homogeneous())
        self.assertTrue(shape_exc.is_homogeneous())
        self.assertFalse(shape_V_m.is_homogeneous(shapes=[shape_inh, shape_exc]))

        shape_V_m = Shape.from_ode("V_m", "-V_m/Tau + (I_in + I_ex) / C_m", initial_values={"V_m": "0."})
        self.assertTrue(shape_V_m.is_homogeneous(shapes=[shape_inh, shape_exc]))
        self.assertFalse(shape_V_m.is_homogeneous(shapes=[]))


if __name__ == '__main__':
    unittest.main()
