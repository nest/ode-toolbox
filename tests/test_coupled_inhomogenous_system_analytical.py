# test_coupled_inhomogenous_system_analytical.py
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

from tests.test_utils import load_test_json
from .context import odetoolbox


def test_coupled_inhomogeneous_system_is_analytical():
    """
    This test is created after the PR #107, in which we removed some conservative numerical checks, leading to amat being classified as mixed-solver.
    """

    model = load_test_json("amat.json")
    result = odetoolbox.analysis(
        model,
        disable_stiffness_check=False)
    assert len(result) == 1
    solver = result[0]
    assert solver["solver"] == "analytical"      # check solver is analytical
    # ensure there are no numerical solvers present
    assert solver["solver"] != "numerical"

    expected_variables = {
        "V_m",
        "V_th_alpha_1",
        "V_th_alpha_2",
        "V_th_v",
        "V_th_v_aux",
        "refr_t"}     # expected state variables from amat
    assert set(solver["state_variables"]) == expected_variables
