#
# test_linearity_checker.py
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


class TestLinearityChecker(unittest.TestCase):

    def test_linear(self):
        shape_inh = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = Shape.from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shape_V_m = Shape.from_ode("V_m", "-V_m/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."})
        shapes = [shape_inh, shape_exc, shape_V_m]
        for shape in shapes:
            self.assertTrue(shape.is_lin_const_coeff())
            self.assertTrue(shape.is_lin_const_coeff(shapes=shapes))


    def test_non_linear(self):
        shape_inh = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = Shape.from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shape_V_m = Shape.from_ode("V_m", "-V_m**2/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."})
        shape_V_m_alt = Shape.from_ode("V_m", "-I_in*V_m/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."})
        shapes = [shape_inh, shape_exc, shape_V_m]
        for shape in [shape_inh, shape_exc]:
            self.assertTrue(shape.is_lin_const_coeff())
            self.assertTrue(shape.is_lin_const_coeff(shapes=shapes))
        self.assertFalse(shape_V_m.is_lin_const_coeff())
        self.assertFalse(shape_V_m.is_lin_const_coeff(shapes=shapes))
        self.assertTrue(shape_V_m_alt.is_lin_const_coeff())		# should be True if is_lin_const_coeff() does not know about the `I_in` symbol
        self.assertFalse(shape_V_m_alt.is_lin_const_coeff(shapes=shapes))		# should be False if is_lin_const_coeff() does know about the `I_in` symbol


if __name__ == '__main__':
    unittest.main()
