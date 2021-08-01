#
# test_lin_const_coeff_and_homogeneous.py
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

import sympy
import unittest

from odetoolbox.shapes import Shape


class TestLinConstCoeffAndHomogeneous(unittest.TestCase):
    """Test homogeneous and linear-and-constant-coefficient judgements on individual ODEs"""
    def setup_method(self, method):
        """pytest "constructor"/setup method"""
        self._parameters = {sympy.Symbol("a"): "1",
                            sympy.Symbol("b"): "3.14159"}


    def test_from_function(self):
        shape = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")

        self.assertTrue(shape.is_homogeneous())
        self.assertTrue(shape.is_lin_const_coeff())
        self.assertTrue(shape.is_lin_const_coeff_in([sympy.Symbol("I_in"), sympy.Symbol("I_in__d")], parameters={sympy.Symbol("tau_syn_in"): "3.14159"}))


    def test_nonlinear_inhomogeneous(self):
        shape = Shape.from_ode("q", "(a - q**2) / b", initial_values={"q": "0."}, parameters=self._parameters)

        self.assertFalse(shape.is_homogeneous())
        self.assertFalse(shape.is_lin_const_coeff())
        self.assertFalse(shape.is_lin_const_coeff_in([sympy.Symbol("q")], parameters=self._parameters))


    def test_nonlinear_homogeneous(self):
        shape = Shape.from_ode("q", "-q**2 / b", initial_values={"q": "0."}, parameters=self._parameters)

        self.assertTrue(shape.is_homogeneous())
        self.assertFalse(shape.is_lin_const_coeff())
        self.assertFalse(shape.is_lin_const_coeff_in([sympy.Symbol("q")], parameters=self._parameters))


    def test_from_homogeneous_ode(self):
        shape = Shape.from_ode("q", "-q / b", initial_values={"q": "0."})

        self.assertTrue(shape.is_homogeneous())
        self.assertFalse(shape.is_lin_const_coeff())
        self.assertTrue(shape.is_lin_const_coeff_in([sympy.Symbol("q")], parameters=self._parameters))


    def test_from_homogeneous_ode(self):
        shape = Shape.from_ode("q", "(a - q) / b", initial_values={"q": "0."}, parameters=self._parameters)
        self.assertFalse(shape.is_homogeneous())
        self.assertTrue(shape.is_lin_const_coeff())
        self.assertTrue(shape.is_lin_const_coeff_in([sympy.Symbol("q")], parameters=self._parameters))

        # xfail case: forgot to specify parameters
        shape = Shape.from_ode("q", "(a - q) / b", initial_values={"q": "0."})
        self.assertTrue(shape.is_homogeneous())
        self.assertFalse(shape.is_lin_const_coeff())


class TestLinConstCoeffAndHomogeneousSystem(unittest.TestCase):
    """Test homogeneous and linear-and-constant-coefficient judgements on systems of ODEs"""

    def setup_method(self, method):
        """pytest "constructor"/setup method"""
        self._parameters = {sympy.Symbol("I_e"): "1.",
                            sympy.Symbol("Tau"): "1.",
                            sympy.Symbol("C_m"): "1.",
                            sympy.Symbol("tau_syn_in"): "1.",
                            sympy.Symbol("tau_syn_ex"): "1."}


    def test_system_of_equations(self):
        all_symbols = [sympy.Symbol(n) for n in ["I_in", "I_in__d", "I_ex", "I_ex__d", "V_m"]]

        shape_inh = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = Shape.from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shape_V_m_lin = Shape.from_ode("V_m", "-V_m/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."}, parameters=self._parameters)
        shape_V_m_lin_no_param = Shape.from_ode("V_m", "-V_m/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."})
        shape_V_m_nonlin = Shape.from_ode("V_m", "-V_m**2/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."}, parameters=self._parameters)
        shape_V_m_nonlin_no_param = Shape.from_ode("V_m", "-V_m**2/Tau + (I_in + I_ex + I_e) / C_m", initial_values={"V_m": "0."})

        for shape in [shape_inh, shape_exc]:
            self.assertTrue(shape.is_lin_const_coeff())
            self.assertTrue(shape.is_homogeneous())
        self.assertTrue(shape.is_lin_const_coeff())
        self.assertTrue(shape.is_homogeneous())

        shapes = [shape_inh, shape_exc, shape_V_m_lin]

        for shape in shapes:
            self.assertTrue(shape_V_m_lin.is_lin_const_coeff_in(symbols=all_symbols, parameters=self._parameters))
            self.assertTrue(shape_V_m_lin_no_param.is_lin_const_coeff_in(symbols=all_symbols, parameters=self._parameters))
            self.assertFalse(shape_V_m_lin_no_param.is_lin_const_coeff_in(symbols=all_symbols))  # xfail when no parameters are specified

            self.assertFalse(shape_V_m_nonlin.is_lin_const_coeff_in(symbols=all_symbols, parameters=self._parameters))
            self.assertFalse(shape_V_m_nonlin_no_param.is_lin_const_coeff_in(symbols=all_symbols, parameters=self._parameters))
            self.assertFalse(shape_V_m_nonlin_no_param.is_lin_const_coeff_in(symbols=all_symbols))


if __name__ == '__main__':
    unittest.main()
