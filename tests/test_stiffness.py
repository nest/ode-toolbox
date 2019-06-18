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

import json
import os
import unittest

try:
    #from .context import odetoolbox
    import odetoolbox
    from odetoolbox.stiffness import StiffnessTester
    HAVE_STIFFNESS = True
except ImportError:
    print("No stiffness")
    HAVE_STIFFNESS = False


def open_json(filename):
    absolute_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
    with open(absolute_filename) as infile:
        indict = json.load(infile)
    return indict


@unittest.skipIf(not HAVE_STIFFNESS,
                 "Stiffness tests not supported on this system")
class TestStiffnessChecker(unittest.TestCase):

    def test_canonical_stiff_system(self):
        indict = open_json("stiff_system.json")

        indict["parameters"]["a"] = "-100"
        result = odetoolbox.analysis(indict, disable_analytic_solver=True)
        assert len(result) == 1 \
         and result[0]["solver"].endswith("implicit")

        indict["parameters"]["a"] = "-1"
        result = odetoolbox.analysis(indict, disable_analytic_solver=True)
        assert len(result) == 1 \
         and result[0]["solver"].endswith("explicit")


    def test_morris_lecar_stiff(self):
        indict = open_json("morris_lecar.json")

        indict["options"]["integration_accuracy"] = 1E-9
        result = odetoolbox.analysis(indict, disable_analytic_solver=True)
        assert len(result) == 1 \
         and result[0]["solver"].endswith("implicit")

        indict["options"]["integration_accuracy"] = 1E-3
        result = odetoolbox.analysis(indict, disable_analytic_solver=True)
        assert len(result) == 1 \
         and result[0]["solver"].endswith("explicit")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
