#
# test_ode_analyzer.py
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
from odetoolbox.analytic import Propagator
from odetoolbox.shapes import Shape


def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestSolutionComputation(unittest.TestCase):

    def test_linearity_checker(self):
        shape_inh = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = Shape.from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shapes = [shape_inh, shape_exc]

        ode_symbol = "V_m"
        ode_definition = "-V_m/Tau + (I_in + I_ex + I_e) / C_m"
        result = odetoolbox.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes)
        self.assertTrue(result)

        ode_symbol = "V_m"
        ode_definition = "(I_in*V_m)/C_m"
        result = odetoolbox.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes)
        self.assertFalse(result)                         

        ode_symbol = "V_m"
        ode_definition = "(V_m*V_m)/C_m"
        result = odetoolbox.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes)
        self.assertFalse(result)


    def test_propagator_matrix(self):
        shape_inh = Shape.from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = Shape.from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shapes = [shape_inh, shape_exc]

        ode_symbol = "V_m"
        ode_definition = "-V_m/Tau + (I_in + I_ex + I_e) / C_m"

        propagator = Propagator(ode_symbol, ode_definition, shapes)

        self.assertEqual(len(propagator.propagator_matrices), 2)  # one per shape
        self.assertTrue(len(propagator.ode_updates) > 0)


    def test_aeif_cond_alpha_implicit(self):
        indict = open_json("aeif_cond_alpha_implicit.json")
        result = odetoolbox.analysis(indict)

        self.assertEqual("analytical", result["solver"])
        self.assertTrue(len(result["propagator"]) > 0)


    def test_iaf_psc_alpha(self):
        indict = open_json("iaf_psc_alpha.json")
        result = odetoolbox.analysis(indict)

        self.assertEqual("analytical", result["solver"])
        self.assertTrue(len(result["propagator"]) > 0)


    def test_iaf_psc_alpha_mixed(self):
        indict = open_json("iaf_psc_alpha_mixed.json")
        result = odetoolbox.analysis(indict)

        self.assertEqual("analytical", result["solver"])
        self.assertTrue(len(result["propagator"]) > 0)


    def test_iaf_cond_alpha(self):
        indict = open_json("iaf_cond_alpha.json")
        result = odetoolbox.analysis(indict)

        solver = result["solver"].split("-")
        self.assertEqual("numeric", solver[0])
        if odetoolbox.HAVE_STIFFNESS:
            self.assertEqual("explicit", solver[1])
        self.assertTrue(len(result["shape_initial_values"]) == 4)
        self.assertTrue(len(result["shape_ode_definitions"]) == 2)
        self.assertTrue(len(result["shape_state_variables"]) == 4)


    def test_iaf_cond_alpha_mixed(self):
        indict = open_json("iaf_cond_alpha_mixed.json")
        result = odetoolbox.analysis(indict)

        solver = result["solver"].split("-")
        self.assertEqual("numeric", solver[0])
        if odetoolbox.HAVE_STIFFNESS:
            self.assertEqual("explicit", solver[1])
        self.assertTrue(len(result["shape_initial_values"]) == 4)
        self.assertTrue(len(result["shape_ode_definitions"]) == 2)
        self.assertTrue(len(result["shape_state_variables"]) == 4)

    def test_shapes_only(self):
        indict = open_json("shapes_only.json")
        result = odetoolbox.analysis(indict)

        self.assertTrue(len(result["shape_initial_values"]) == 4)
        self.assertTrue(len(result["shape_ode_definitions"]) == 2)
        self.assertTrue(len(result["shape_state_variables"]) == 4)

if __name__ == '__main__':
    unittest.main()
