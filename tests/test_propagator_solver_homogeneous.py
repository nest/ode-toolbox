#
# test_analytic_solver_integration.py
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

from .context import odetoolbox
from tests.test_utils import _open_json


class TestPropagatorSolverHomogeneous:
    r"""Test ODE-toolbox ability to come up with a propagator solver for a matrix that is not block-diagonalisable, because it contains an autonomous ODE."""

    def test_propagator_solver_homogeneous(self):
        indict = _open_json("test_propagator_solver_homogeneous.json")
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True, log_level="DEBUG")
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"
        assert float(solver_dict["propagators"]["__P__refr_t__refr_t"]) == 1.
        assert solver_dict["propagators"]["__P__V_m__V_m"] == "1.0*exp(-__h/tau_m)"
