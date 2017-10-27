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

from .context import odetoolbox

try:
    from odetoolbox.stiffness import check_ode_system_for_stiffness
    HAVE_STIFFNESS = True
except:
    HAVE_STIFFNESS = False

def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict

@unittest.skipIf(not HAVE_STIFFNESS,
                 "Stiffness tests not supported on this system")
class TestStiffnessChecker(unittest.TestCase):

    def test_iaf_cond_alpha_stiff(self):
        indict = open_json("iaf_cond_alpha_odes_stiff.json")
        result = check_ode_system_for_stiffness(indict)
        self.assertEquals("implicit", result)


    def test_iaf_cond_alpha_odes(self):
        indict = open_json("iaf_cond_alpha_odes.json")
        result = check_ode_system_for_stiffness(indict)
        self.assertEquals("explicit", result)

            
    def test_iaf_cond_alpha_odes_threshold(self):
        indict = open_json("iaf_cond_alpha_odes_threshold.json")
        result = check_ode_system_for_stiffness(indict)
        self.assertEquals("explicit", result)


if __name__ == '__main__':
    unittest.main()
