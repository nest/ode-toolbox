#
# test_singularity_detection.py
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
import pytest

from odetoolbox.singularity_detection import SingularityDetection


class TestSingularityDetection:
    r"""Test singularity detection"""

    def test_is_matrix_defined_under_substitution(self):
        tau_m, tau_r, C, h = sympy.symbols("tau_m, tau_r, C, h")
        P = sympy.Matrix([[-1 / tau_r, 0, 0], [1, -1 / tau_r, 0], [0, 1 / C, -1 / tau_m]])
        assert SingularityDetection._is_matrix_defined_under_substitution(P, {})
        assert SingularityDetection._is_matrix_defined_under_substitution(P, {tau_r: 1})
        assert not SingularityDetection._is_matrix_defined_under_substitution(P, {tau_r: 0})

    @pytest.mark.parametrize("kernel_to_use", ["alpha", "beta"])
    def test_alpha_beta_kernels(self, kernel_to_use: str):
        r"""Test correctness of result for simple leaky integrate-and-fire neuron with biexponential postsynaptic kernel"""
        if kernel_to_use == "alpha":
            tau_m, tau_s, C, h = sympy.symbols("tau_m, tau_s, C, h")
            A = sympy.Matrix([[-1 / tau_s, 0, 0], [1, -1 / tau_s, 0], [0, 1 / C, -1 / tau_m]])
        elif kernel_to_use == "beta":
            tau_m, tau_d, tau_r, C, h = sympy.symbols("tau_m, tau_d, tau_r, C, h")
            A = sympy.Matrix([[-1 / tau_d, 0, 0], [1, -1 / tau_r, 0], [0, 1 / C, -1 / tau_m]])

        P = sympy.simplify(sympy.exp(A * h))  # Propagator matrix

        condition = SingularityDetection._generate_singularity_conditions(P)
        condition = SingularityDetection._flatten_conditions(condition)  # makes a list of conditions with each condition in the form of a dict
        condition = SingularityDetection._filter_valid_conditions(condition, A)  # filters out the invalid conditions (invalid means those for which A is not defined)

        if kernel_to_use == "alpha":
            assert len(condition) == 1
        elif kernel_to_use == "beta":
            assert len(condition) == 3

    def test_more_than_one_solution(self):
        r"""Test the case where there is more than one element returned in a solution to an equation; in this example, for a quadratic input equation"""
        A = sympy.Matrix([[sympy.parsing.sympy_parser.parse_expr("-1/(tau_s**2 - 3*tau_s - 42)")]])
        condition = SingularityDetection._generate_singularity_conditions(A)
        assert len(condition) == 2
        for cond in condition:
            assert sympy.Symbol("tau_s") in cond.keys()
        assert cond[sympy.Symbol("tau_s")] == sympy.parsing.sympy_parser.parse_expr("3/2 + sqrt(177)/2") \
               or cond[sympy.Symbol("tau_s")] == sympy.parsing.sympy_parser.parse_expr("3/2 - sqrt(177)/2")
