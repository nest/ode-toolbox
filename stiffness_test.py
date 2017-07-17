import unittest
from stiffness import *


class TestStiffnessChecker(unittest.TestCase):

    def test_range_with_iaf_cond_alpha(self):
        threshold_body = "y_0 >= V_th"

        start = 0.03
        stop = 1.0
        step = 0.01

        result = {}
        current_step = start
        while current_step <= stop:
            default_values = ["neuron_name = 'iaf_cond_alpha'",
                              "tau_synE = {}".format(current_step),  # "tau_synE = 0.2",
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
                              "initial_values = [0, e / tau_synE, 0, -e / tau_synI, 0]"]

            odes = ["function I_leak = -(g_L * (y_0 - E_L))",
                    "f_0 = (I_leak - (y_2 * (y_0 - E_ex)) - y_4 * (y_0 - E_in) + I_e) / C_m",
                    "f_1 = -y_1 / tau_synE",
                    "f_2 = y_1 - (y_2 / tau_synE)",
                    "f_3 = -y_3 / tau_synI",
                    "f_4 = y_3 - (y_4 / tau_synI)"]

            method = check_ode_system_for_stiffness(odes, default_values, threshold_body)
            result[current_step] = method
            print current_step, method
            current_step += step

    def test_with_iaf_cond_alpha(self):
        threshold_body = "y_0 >= V_th"

        default_values = ["neuron_name = 'iaf_cond_alpha'",
                          "tau_synE = 0.05",  # "tau_synE = 0.2",
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
                          "initial_values = [0, e / tau_synE, 0, -e / tau_synI, 0]"]

        odes = ["function I_leak = -(g_L * (y_0 - E_L))",
                "f_0 = (I_leak - (y_2 * (y_0 - E_ex)) - y_4 * (y_0 - E_in) + I_e) / C_m",
                "f_1 = -y_1 / tau_synE",
                "f_2 = y_1 - (y_2 / tau_synE)",
                "f_3 = -y_3 / tau_synI",
                "f_4 = y_3 - (y_4 / tau_synI)"]

        print check_ode_system_for_stiffness(odes, default_values, threshold_body)

    def test_iaf_neuron(self):
        threshold_body = "y_0 >= Theta"

        default_values = ["neuron_name = 'iaf_neuron'",
                          "tau_syn_in = 2.",
                          "tau_syn_ex = 0.05",
                          "V_m = -70.",
                          "C_m = 250.",
                          "E_L = -70.",
                          "Tau = 10.",
                          "Theta = -55.",
                          "start_values = [E_L, 0.0, 0.0, 0.0, 0.0]",
                          "initial_values = [0, e / tau_syn_ex, 0, -e / tau_syn_in, 0]"]

        odes = ["f_0 = -y_0 / Tau + (y_2 + y_4)/ C_m",
                "f_1 = -y_1 / tau_syn_ex",
                "f_2 = y_1 - (y_2 / tau_syn_ex)",
                "f_3 = -y_3 / tau_syn_in",
                "f_4 = y_3 - (y_4 / tau_syn_in)"]

        print check_ode_system_for_stiffness(odes, default_values, threshold_body)

    def test_aeif_cond_alpha(self):
        threshold_body = "y_0 >= V_th"

        default_values =["name = 'aeif_cond_alpha'",
                         "tau_synE = 0.05",
                         "tau_synI = 2.0",
                         "E_ex = 0.0",
                         "E_in = -85.0",
                         "g_L = 30.0",
                         "I_e = 0.0",
                         "E_L = -70.6",
                         "C_m = 281.0",
                         "V_th = -50.4",
                         "V_peak = 0.0",
                         "tau_w = 144.0",
                         "a = 4.0",
                         "Delta_T = 2.0",
                         "start_values = [E_L, 0.0, 0.0, 0.0]",
                         "initial_values = [0, 1, -1, 0]"]
                         
        odes = ["function I_syn_exc = y_1 * (y_0 - E_ex)",
                "function I_syn_inh = y_2 * (y_0 - E_in)",
                "function I_spike = g_L * Delta_T * exp((y_0 - V_th) / Delta_T)",
                "f_0 = (g_L * (y_0 - E_L) + I_spike - I_syn_exc - I_syn_inh - y_3 + I_e) / C_m",
                "f_1 = -y_1 / tau_synE",
                "f_2 = -y_2 / tau_synI",
                "f_3 = (a * (y_0 - E_L) - y_3) / tau_w"]

        print check_ode_system_for_stiffness(odes, default_values, threshold_body)

    def test_range_with_hh_iaf_psc_alpha(self):
        threshold_body = "false"

        start = 0.11
        stop = 0.14
        step = 0.01

        result = {}
        current_step = start
        while current_step <= stop:
            default_values = ["g_Na = 12000.0",
                              "g_K = 3600.0",
                              "g_L = 30.0",
                              "C_m = 100.0",
                              "E_Na = 50.0",
                              "E_K = -77.0",
                              "E_L = -54.402",
                              "tau_synE = 0.05",
                              "tau_synI = 2.0",
                              "I_e = 0.0",
                              "I_stim = 0.0",
                              "start_values = [E_L, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
                              "initial_values = [0, 0, 0, 0, e / tau_synE, 0, -e / tau_synI, 0]"]

            odes = ["function V = y_0",
                    "function m = y_1",
                    "function h = y_2",
                    "function n = y_3",
                    "function dI_ex = y_4",
                    "function I_ex = y_5",
                    "function dI_in = y_6",
                    "function I_in = y_7",
                    "function alpha_n =  ( 0.01 * ( V + 55. ) ) / ( 1. - exp( -( V + 55. ) / 10. ) )",
                    "function beta_n = 0.125 * exp( -( V + 65. ) / 80. )",
                    "function alpha_m = ( 0.1 * ( V + 40. ) ) / ( 1. -exp( -( V + 40. ) / 10. ) )",
                    "function beta_m = 4. * exp( -( V + 65. ) / 18. )",
                    "function alpha_h = 0.07 * exp( -( V + 65. ) / 20. )",
                    "function beta_h = 1. / ( 1. + exp( -( V + 35. ) / 10. ) )",
                    "function I_Na = g_Na * m * m * m * h * ( V - E_Na )",
                    "function I_K = g_K * n * n * n * n * ( V - E_K )",
                    "function I_L = g_L * ( V - E_L )",
                    "f_0 = ( -( I_Na + I_K + I_L ) + I_stim + I_e + I_ex + I_in ) / C_m",
                    "f_1 = alpha_m * ( 1 - m ) - beta_m * m",
                    "f_2 = alpha_h * ( 1 - h ) - beta_h * h",
                    "f_3 = alpha_n * ( 1 - n ) - beta_n * n",
                    "f_4 = -dI_ex / tau_synE",
                    "f_5 = dI_ex - ( I_ex / tau_synE )",
                    "f_6 = -dI_in / tau_synI",
                    "f_7 = dI_in - ( I_in / tau_synI )"]

            method = check_ode_system_for_stiffness(odes, default_values, threshold_body)
            result[current_step] = method
            print current_step, method
            current_step += step

    def test_hh_iaf_psc_alpha(self):
        threshold_body = "false"

        default_values = ["g_Na = 12000.0",
                          "g_K = 3600.0",
                          "g_L = 30.0",
                          "C_m = 100.0",
                          "E_Na = 50.0",
                          "E_K = -77.0",
                          "E_L = -54.402",
                          "tau_synE = 0.05",
                          "tau_synI = 2.0",
                          "I_e = 0.0",
                          "I_stim = 0.0",
                          "start_values = [E_L, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
                          "initial_values = [0, 0, 0, 0, e / tau_synE, 0, -e / tau_synI, 0]"]

        odes = ["function V = y_0",
                "function m = y_1",
                "function h = y_2",
                "function n = y_3",
                "function dI_ex = y_4",
                "function I_ex = y_5",
                "function dI_in = y_6",
                "function I_in = y_7",
                "function alpha_n =  ( 0.01 * ( V + 55. ) ) / ( 1. - exp( -( V + 55. ) / 10. ) )",
                "function beta_n = 0.125 * exp( -( V + 65. ) / 80. )",
                "function alpha_m = ( 0.1 * ( V + 40. ) ) / ( 1. -exp( -( V + 40. ) / 10. ) )",
                "function beta_m = 4. * exp( -( V + 65. ) / 18. )",
                "function alpha_h = 0.07 * exp( -( V + 65. ) / 20. )",
                "function beta_h = 1. / ( 1. + exp( -( V + 35. ) / 10. ) )",
                "function I_Na = g_Na * m * m * m * h * ( V - E_Na )",
                "function I_K = g_K * n * n * n * n * ( V - E_K )",
                "function I_L = g_L * ( V - E_L )",
                "f_0 = ( -( I_Na + I_K + I_L ) + I_stim + I_e + I_ex + I_in ) / C_m",
                "f_1 = alpha_m * ( 1 - m ) - beta_m * m",
                "f_2 = alpha_h * ( 1 - h ) - beta_h * h",
                "f_3 = alpha_n * ( 1 - n ) - beta_n * n",
                "f_4 = -dI_ex / tau_synE",
                "f_5 = dI_ex - ( I_ex / tau_synE )",
                "f_6 = -dI_in / tau_synI",
                "f_7 = dI_in - ( I_in / tau_synI )"]

        print check_ode_system_for_stiffness(odes, default_values, threshold_body)

    def test_with_iaf_cond_alpha_mc(self):
        threshold_body = "y_0 >= V_th"

        default_values = ["neuron_name = 'iaf_cond_alpha_mc'",
                          "V_th = -55.0",
                          "V_reset = -60.0",
                          "E_L_1 = -70.0",
                          "tau_synE_1 = 0.05",
                          "tau_synI_1 = 2.0",
                          "E_ex_1 = 0.0",
                          "E_in_1 = -85.0",
                          "g_L_1 = 1",
                          "I_e_1 = 0.0",
                          "C_m_1 = 150",
                          "E_L_2 = -85.0",
                          "tau_synE_2 = 0.5",
                          "tau_synI_2 = 2.0",
                          "E_ex_2 = 0.0",
                          "E_in_2 = -70.0",
                          "g_L_2 = 500.0",
                          "I_e_2 = 0.0",
                          "C_m_2 = 75.",
                          "E_L_3 = -70.0",
                          "tau_synE_3 = 0.5",
                          "tau_synI_3 = 2.0",
                          "E_ex_3 = 0.0",
                          "E_in_3 = -85.0",
                          "g_L_3 = 10",
                          "I_e_3 = 0.0",
                          "C_m_3 = 150",
                          "start_values = [E_L_1, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, E_L_3, 0.0, 0.0, 0.0, 0.0]",
                          "initial_values = [0, e / tau_synE_1, 0, -e / tau_synI_1, 0, 0, e / tau_synE_2, 0, -e / tau_synI_2, 0, 0, e / tau_synE_3, 0, -e / tau_synI_3, 0]",
                          "conn_1 = 10",
                          "conn_2 = 0."]

        odes = ["f_0 = (-(g_L_1 * (y_0 - E_L_1)) - (y_2 * (y_0 - E_ex_1)) - y_4 * (y_0 - E_in_1) + conn_1 * (y_0 - y_5) + I_e_1) / C_m_1",
                "f_1 = -y_1 / tau_synE_1",
                "f_2 = y_1 - (y_2 / tau_synE_1)",
                "f_3 = -y_3 / tau_synI_1",
                "f_4 = y_3 - (y_4 / tau_synI_1)",
                "f_5 = (-(g_L_2 * (y_5 - E_L_2)) - (y_7 * (y_5 - E_ex_2)) - y_9 * (y_5 - E_in_2) + conn_1 * (y_5 - y_0) + conn_2 * (y_5 - y_10) + I_e_2) / C_m_2",
                "f_6 = -y_6 / tau_synE_2",
                "f_7 = y_6 - (y_7 / tau_synE_2)",
                "f_8 = -y_8 / tau_synI_2",
                "f_9 = y_8 - (y_9 / tau_synI_2)",
                "f_10 = (-(g_L_3 * (y_10 - E_L_3)) - (y_12 * (y_10 - E_ex_3)) - y_14 * (y_10 - E_in_3) + conn_2 * (y_10 -y_5) + I_e_3) / C_m_3",
                "f_11 = -y_11 / tau_synE_3",
                "f_12 = y_11 - (y_12 / tau_synE_3)",
                "f_13= -y_13 / tau_synI_3",
                "f_14 = y_13 - (y_14 / tau_synI_3)"]

        print check_ode_system_for_stiffness(odes, default_values, threshold_body)

    def test_range_with_izhikevich(self):
        threshold_body = "false"

        start = 0.03
        stop = 1.0
        step = 0.01

        result = {}
        current_step = start
        while current_step <= stop:
            default_values = ["neuron_name = 'izhikevich'",
                              "a = 0.02",
                              "b = 0.2",
                              "I_e = 0.0",
                              "I = 0.0",
                              "start_values = [-65., 0.0]",
                              "initial_values = [0, 0]"]

            odes = ["f_0 =  0.04 * y_0 * y_0 + 5.0 * y_0 + ( 140 - y_1 ) + ( (I + I_e) )",
                    "f_1 = a*(b*y_0-y_1)"]

            method = check_ode_system_for_stiffness(odes, default_values, threshold_body)
            result[current_step] = method
            print current_step, method
            current_step += step


if __name__ == '__main__':
    unittest.main()
