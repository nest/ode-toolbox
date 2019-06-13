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

#indict = open_json("iaf_cond_alpha_odes.json")
#result = odetoolbox.analysis(indict, disable_analytic_solver=True)
#assert len(result) == 1 \
    #and result[0]["solver"].endswith("explicit")



@unittest.skipIf(not HAVE_STIFFNESS,
                 "Stiffness tests not supported on this system")
class TestStiffnessChecker(unittest.TestCase):

    #def test_iaf_cond_alpha_odes(self):
        #indict = open_json("iaf_cond_alpha_odes.json")
        #result = odetoolbox.analysis(indict, disable_analytic_solver=True)
        #assert len(result) == 1 \
         #and result[0]["solver"].endswith("explicit")

    """def test_canonical_stiff_system(self):
        indict = open_json("stiff_system.json")
        result = odetoolbox.analysis(indict, disable_analytic_solver=True)
        assert len(result) == 1 \
         and result[0]["solver"].endswith("implicit")"""

    def test_morris_lecar_stiff(self):
        indict = open_json("morris_lecar.json")
        result = odetoolbox.analysis(indict, disable_analytic_solver=True)
        assert len(result) == 1 \
         and result[0]["solver"].endswith("implicit")

        #assert (result[0]["solver"].endswith("implicit") and result[1]["solver"] == "analytical") \
         #or (result[1]["solver"].endswith("implicit") and result[0]["solver"] == "analytical")

"""    def test_iaf_cond_alpha_odes_threshold(self):
        indict = open_json("iaf_cond_alpha_odes_threshold.json")
        tester = StiffnessTester(indict)
        result = tester.check_stiffness()
        self.assertEqual("explicit", result)

    def test_fitzhugh_nagumo(self):
        indict = open_json("fitzhugh_nagumo.json")
        tester = StiffnessTester(indict)
        result = tester.check_stiffness(sim_resolution=0.05, accuracy=1e-5)
        self.assertEqual("explicit", result)

    def test_morris_lecar(self):
        indict = open_json("morris_lecar.json")
        tester = StiffnessTester(indict)
        result = tester.check_stiffness(sim_resolution=0.2, accuracy=1e-5)
        self.assertEqual("explicit", result)

"""

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
