#
# test_stiffness.py
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

import odetoolbox
from tests.test_utils import _open_json

try:
    import pygsl
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False


@pytest.mark.skipif(not PYGSL_AVAILABLE, reason="Cannot run stiffness test if GSL is not installed.")
class TestStiffnessChecker:

    def test_canonical_stiff_system(self):
        indict = _open_json("stiff_system.json")

        indict["parameters"]["a"] = "-100"
        result = odetoolbox.analysis(indict, disable_analytic_solver=True, disable_stiffness_check=not PYGSL_AVAILABLE)
        assert len(result) == 1 \
               and result[0]["solver"].endswith("implicit")

        indict["parameters"]["a"] = "-1"
        result = odetoolbox.analysis(indict, disable_analytic_solver=True, disable_stiffness_check=not PYGSL_AVAILABLE)
        assert len(result) == 1 \
               and result[0]["solver"].endswith("explicit")


    def test_morris_lecar_stiff(self):
        indict = _open_json("morris_lecar.json")

        indict["options"]["integration_accuracy_abs"] = 1E-9
        indict["options"]["integration_accuracy_rel"] = 1E-9
        result = odetoolbox.analysis(indict, disable_analytic_solver=True, disable_stiffness_check=not PYGSL_AVAILABLE)
        assert len(result) == 1 \
               and result[0]["solver"].endswith("implicit")

        indict["options"]["integration_accuracy_abs"] = 1E-3
        indict["options"]["integration_accuracy_rel"] = 1E-3
        result = odetoolbox.analysis(indict, disable_analytic_solver=True, disable_stiffness_check=not PYGSL_AVAILABLE)
        assert len(result) == 1 \
               and result[0]["solver"].endswith("explicit")
