import unittest
from stiffness_arbitrary_input import *


class TestStiffnessChecker(unittest.TestCase):

    def test_with_iaf_cond_alpha(self):
        threshold_body = "y_0 >= V_th"
        reset_statement = "y_0 = E_L"

        default_values = ["neuron_name = 'iaf_cond_alpha'",
                          "tau_synE = 0.2",
                          "tau_synI = 2.0",
                          "E_ex = 0.0",
                          "E_in = -85.0",
                          "g_L = 16.6667",
                          "I_e = 0.0",
                          "E_L = -70.0",
                          "C_m = 250",
                          "V_th = -55.0",
                          "I_stim = 0.",
                          "start_values = [E_L, 0.0, 0.0, 0.0, 0.0]",
                          "initial_values = [0, math.e / tau_synE, 0, -math.e / tau_synI, 0]"]

        odes = ["f_0 = (-(g_L * (y_0 - E_L)) - (y_2 * (y_0 - E_ex)) - y_4 * (y_0 - E_in) + I_e) / C_m",
                "f_1 = -y_1 / tau_synE",
                "f_2 = y_1 - (y_2 / tau_synE)",
                "f_3 = -y_3 / tau_synI",
                "f_4 = y_3 - (y_4 / tau_synI)"]

        check_ode_system_for_stiffness(odes, default_values, threshold_body, reset_statement)

    def test_iaf_neuron(self):
        threshold_body = "y_0 >= Theta"
        reset_statement = "y_0 = E_L"

        default_values = ["neuron_name = 'iaf_neuron'",
                          "tau_syn_in = 2.",
                          "V_m = -70.",
                          "C_m = 250.",
                          "E_L = -70.",
                          "Tau = 10.",
                          "Theta = -55.",
                          "start_values = [-70., 0., 0.]",
                          "initial_values = [math.e / tau_syn_in, 0, 0]"]

        odes = ["f_0 = -y_0/Tau + y_2 / C_m",
                "f_1 = -y_1 / tau_syn_in",
                "f_2 = y_1 - (y_2 / tau_syn_in)"]

        # TODO runs to slow check_ode_system_for_stiffness(odes, default_values, threshold_body, reset_statement)
        
        
if __name__ == '__main__':
    unittest.main()
