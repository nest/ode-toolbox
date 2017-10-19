import json
import unittest

import ode_analyzer
from prop_matrix import Propagator
from shapes import shape_from_function


class TestSolutionComputation(unittest.TestCase):

    def test_linearity_checker(self):
        shape_inh = shape_from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = shape_from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        shapes = [shape_inh, shape_exc]

        ode_symbol = "V_m"
        ode_definition = "-V_m/Tau + (I_in + I_ex + I_e) / C_m"

        self.assertEqual(True,
                         ode_analyzer.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes))

        ode_symbol = "V_m"
        ode_definition = "(I_in*V_m)/C_m"
        self.assertEqual(False,
                         ode_analyzer.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes))

        ode_symbol = "V_m"
        ode_definition = "(V_m*V_m)/C_m"
        self.assertEqual(False,
                         ode_analyzer.ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes))

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
        result = ode_analyzer.main(["iaf_psc_alpha.json"])
        result = json.loads(result)

        self.assertEqual("exact", result["solver"])
        self.assertTrue(len(result["propagator"]) > 0)

    def test_iaf_cond_alpha(self):
        result = ode_analyzer.main(["iaf_cond_alpha.json"])
        result = json.loads(result)

        self.assertEqual("numeric", result["solver"])
        self.assertTrue(len(result["shape_initial_values"]) == 4)
        self.assertTrue(len(result["shape_ode_definitions"]) == 2)
        self.assertTrue(len(result["shape_state_variables"]) == 4)

    def test_iaf_cond_alpha_mixed(self):
        result = ode_analyzer.main(["iaf_cond_alpha_mixed.json"])
        result = json.loads(result)

        self.assertEqual("numeric", result["solver"])
        self.assertTrue(len(result["shape_initial_values"]) == 4)
        self.assertTrue(len(result["shape_ode_definitions"]) == 2)
        self.assertTrue(len(result["shape_state_variables"]) == 4)


if __name__ == '__main__':
    unittest.main()
