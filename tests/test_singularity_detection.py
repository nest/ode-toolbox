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

import io
import logging
import sympy
import pytest

from .context import odetoolbox
from tests.test_utils import _open_json
from odetoolbox.singularity_detection import SingularityDetection
from odetoolbox.sympy_helpers import SymmetricEq


class TestSingularityDetection:
    r"""Test singularity detection"""

    def test_is_matrix_defined_under_substitution(self):
        tau_m, tau_r, C, h = sympy.symbols("tau_m, tau_r, C, h")
        P = sympy.Matrix([[-1 / tau_r, 0, 0], [1, -1 / tau_r, 0], [0, 1 / C, -1 / tau_m]])
        assert SingularityDetection._is_matrix_defined_under_substitution(P, set())
        assert SingularityDetection._is_matrix_defined_under_substitution(P, set([SymmetricEq(tau_r, 1)]))
        assert not SingularityDetection._is_matrix_defined_under_substitution(P, set([SymmetricEq(tau_r, 0)]))

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
        print(condition)
        condition = SingularityDetection._filter_valid_conditions(condition, A)  # filters out the invalid conditions (invalid means those for which A is not defined)

        if kernel_to_use == "alpha":
            assert len(condition) == 1
        elif kernel_to_use == "beta":
            assert len(condition) == 3

    def test_more_than_one_solution(self):
        r"""Test the case where there is more than one element returned in a solution to an equation; in this example, for a quadratic input equation"""
        A = sympy.Matrix([[sympy.parsing.sympy_parser.parse_expr("-1/(tau_s**2 - 3*tau_s - 42)")]])
        conditions = SingularityDetection._generate_singularity_conditions(A)
        assert len(conditions) == 2
        for cond_set in conditions:
            for cond in cond_set:
                assert sympy.Symbol("tau_s") == cond.lhs
                assert cond.rhs == sympy.parsing.sympy_parser.parse_expr("3/2 + sqrt(177)/2") \
                    or cond.rhs == sympy.parsing.sympy_parser.parse_expr("3/2 - sqrt(177)/2")


class TestPropagatorSolverHomogeneous:
    r"""Test ODE-toolbox ability to ignore imaginary roots of the dynamical equations.

    This dynamical system is difficult for sympy to solve and it needs to do quite some number crunching.

    Test that no singularity conditions are found for this system after analysis.
    """

    def test_propagator_solver_homogeneous(self):
        indict = {"dynamics": [{"expression": "V_m' = (1.0E+03 * ((V_m * 1.0E+03) / ((tau_m * 1.0E+15) * (1 + exp(alpha_exp * (V_m_init * 1.0E+03))))))",
                                "initial_values": {"V_m": "(1.0E+03 * (-70.0 * 1.0E-03))"}}],
                  "options": {"output_timestep_symbol": "__h",
                              "simplify_expression": "sympy.logcombine(sympy.powsimp(sympy.expand(expr)))"},
                  "parameters": {"V_m_init": "(1.0E+03 * (-65.0 * 1.0E-03))",
                                 "alpha_exp": "2 / ((3.0 * 1.0E+06))",
                                 "tau_m": "(1.0E+15 * (2.0 * 1.0E-03))"}}

        logger = logging.getLogger()

        log_stream = io.StringIO()
        log_handler = logging.StreamHandler(log_stream)
        logger.addHandler(log_handler)

        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True, log_level="DEBUG")

        log_contents = log_stream.getvalue()

        # test that no singularity conditions were found for this system
        assert not "Under certain conditions" in log_contents
        assert not "division by zero" in log_contents
