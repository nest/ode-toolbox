import glob
import unittest

import os
from pandas import json

from prop_matrix import Propagator
from shapes import *
import OdeAnalyzer


class TestSolutionComputation(unittest.TestCase):

    def setUp(self):
        files = glob.glob('*.json')
        for f in files:
            os.remove(f)

    def test_linearity_checker(self):
        shape_inh = shape_from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = shape_from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shapes = [shape_inh, shape_exc]

        ode_symbol = "V_m"
        ode_definition = "-V_m/Tau + (I_in + I_ex + I_e) / C_m"

        self.assertEqual(True,
                         OdeAnalyzer.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes))

        ode_symbol = "V_m"
        ode_definition = "(I_in*V_m)/C_m"
        self.assertEqual(False,
                         OdeAnalyzer.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes))

        ode_symbol = "V_m"
        ode_definition = "(V_m*V_m)/C_m"
        self.assertEqual(False,
                         OdeAnalyzer.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes))

    def test_propagator_matrix(self):
        shape_inh = shape_from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = shape_from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shapes = [shape_inh, shape_exc]

        ode_symbol = "V_m"
        ode_definition = "-V_m/Tau + (I_in + I_ex + I_e) / C_m"

        propagator = Propagator(ode_symbol, ode_definition, shapes)

        self.assertTrue(len(propagator.propagator_matrices) == 2)  # one matrix for every shape
        self.assertTrue(len(propagator.ode_updates) > 0)

    def test_iaf_psc_alpha(self):
        OdeAnalyzer.main(["test/iaf_psc_alpha.json"])
        files = glob.glob('*.json')
        self.assertTrue(len(files) == 1)
        result = json.load(open(files[0]))

        self.assertEqual("exact", result["solver"])
        self.assertTrue(len(result["propagator"]) > 0)

    def test_iaf_cond_alpha(self):
        OdeAnalyzer.main(["test/iaf_cond_alpha.json"])
        files = glob.glob('*.json')
        self.assertTrue(len(files) == 1)
        result = json.load(open(files[0]))

        self.assertEqual("numeric", result["solver"])
        self.assertTrue(len(result["shape_initial_values"]) == 4)
        self.assertTrue(len(result["shape_ode_definitions"]) == 2)
        self.assertTrue(len(result["shape_state_variables"]) == 4)


if __name__ == '__main__':
    unittest.main()