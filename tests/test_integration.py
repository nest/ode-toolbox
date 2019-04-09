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
import sympy
import numpy as np

import matplotlib.pyplot as plt

from .context import odetoolbox
from odetoolbox.analytic import Propagator
from odetoolbox.shapes import shape_from_function

from math import e
from sympy import exp, sympify
            
def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestSolutionComputation(unittest.TestCase):
    '''
    Test JSON:
    
    {
        "shapes": [
            {
            "type": "function",
            "symbol": "I_shape_in",
            "definition": "(e/tau_syn_in) * t * exp(-t/tau_syn_in)"
            },
            {
            "type": "function",
            "symbol": "I_shape_ex",
            "definition": "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)"
            }
        ],

        "odes": [
            {
            "symbol": "V_abs",
            "definition": "(-1)/Tau*V_abs+1/C_m*(I_shape_in+I_shape_ex+I_e+currents)"
            }
        ],

        "parameters": {}
    }
    '''

    def test_iaf_psc_alpha(self):
        
        debug = True
        
        indict = open_json("iaf_psc_alpha.json")
        result = odetoolbox.analysis(indict)

        assert result["solver"] == "analytical"

        h = 2.5E-4    # [s]
        T = 44E-3#0E-3    # [s] XXX TODO
        
        # neuron parameters
        tau = 20E-3    # [s]
        tau_syn = 5E-3    # [s]
        c_m = 1E-6    # [F]
        v_abs_init = -70E-3   # [V]
        i_ex_init = [0., 10E-6]   # [A]
        spike_time = 200E-3 # [s]
        
        N = int(np.ceil(T / h))
        t = np.linspace(0., T, N)
        spike_time_idx = np.argmin((spike_time - t)**2)
        N_shapes = len(result["shape_state_variables"])


        #
        #   forward Euler reference timeseries
        #

        v_abs = np.empty(N)
        v_abs[0] = v_abs_init
        #I_shape_ex_expr_idx = np.where([indict["shapes"][i]["symbol"] == "I_shape_ex" for i in range(len(indict["shapes"]))])[0][0]
        #I_shape_ex_expr = eval(indict["shapes"][I_shape_ex_expr_idx]["definition"])
        #I_shape_ex = shape_from_function(indict["shapes"][I_shape_ex_expr_idx]["symbol"], indict["shapes"][I_shape_ex_expr_idx]["definition"])
        i_ex = np.zeros((2, N))
        #i_ex[:, 0] = i_ex_init
        for step in range(1, N):
            if step - 1 == spike_time_idx:
                i_ex[:, step - 1] = i_ex_init
            '''V_abs_expr_ = V_abs_expr.copy()
            V_abs_expr_ = V_abs_expr_.subs(Tau, tau)
            V_abs_expr_ = V_abs_expr_.subs(C_m, c_m)
            V_abs_expr_ = V_abs_expr_.subs(V_abs, v_abs[step - 1])
            V_abs_expr_ = V_abs_expr_.subs(I_e, 0.)
            V_abs_expr_ = V_abs_expr_.subs(currents, 0.)
            V_abs_expr_ = V_abs_expr_.subs(I_shape_in, 0.)
            V_abs_expr_ = V_abs_expr_.subs(I_shape_ex, 0.)#i_ex[0, step - 1])'''
            
            V_abs_expr_ = -1/tau*v_abs[step - 1]+1/c_m*(i_ex[0, step - 1])
            v_abs[step] = v_abs[step - 1] + h * V_abs_expr_
            
            i_ex[:, step] = np.array(i_ex[:, step - 1]) + h * np.array([i_ex[1, step - 1], -i_ex[0, step - 1] / tau_syn**2 - 2 * i_ex[1, step - 1] / tau_syn])


        #
        #   timeseries using ode-toolbox generated propagators
        #

        V_abs, Tau, Tau_syn_in, Tau_syn_ex, C_m,  I_e, currents = sympy.symbols("V_abs Tau Tau_syn_in Tau_syn_ex C_m  I_e currents")
        for shape_idx in range(N_shapes):
            for shape_variable_idx, (shape_variable_name, shape_initial_value, shape_state_update) in enumerate(zip(result["shape_state_variables"][shape_idx], result["shape_initial_values"][shape_idx], result["shape_state_updates"][shape_idx])):
                print("Defining var: " + str(shape_variable_name + " = sympy.symbols(\"" + shape_variable_name + "\")"))
                exec(shape_variable_name + " = sympy.symbols(\"" + shape_variable_name + "\")", globals())



        N_shape_variables = np.sum(len(shape_state_variable_names) for shape_state_variable_names in result["shape_state_variables"])

        shape_state = {}
        for shape_idx in range(N_shapes):
            shape_name = result["shape_state_variables"][shape_idx][0]
            shape_state[shape_name] = {}
            dim = len(result["shape_state_variables"])
            for shape_variable_idx, (shape_variable_name, shape_initial_value, shape_state_update) in enumerate(zip(result["shape_state_variables"][shape_idx], result["shape_initial_values"][shape_idx], result["shape_state_updates"][shape_idx])):
                expr = sympify(eval(shape_initial_value.replace("__h", "h")))
                expr = expr.subs(Tau, tau)
                expr = expr.subs(C_m, c_m)
                expr = expr.subs(Tau_syn_in, tau_syn)
                expr = expr.subs(Tau_syn_ex, tau_syn)
                
                shape_state[shape_name][shape_variable_name] = np.zeros(N)
                shape_state[shape_name][shape_variable_name][0] = expr

        V_abs_update_expr_str = result["ode_updates"]["V_abs"]["constant"]
        V_abs_update_expr = eval(V_abs_update_expr_str.replace("__h", "h"))
        V_abs_update_expr = V_abs_update_expr.subs(Tau, tau)
        V_abs_update_expr = V_abs_update_expr.subs(C_m, c_m)
        V_abs_update_expr = V_abs_update_expr.subs(currents, 0.)
        V_abs_update_expr = V_abs_update_expr.subs(I_shape_in, 0.)

        v_abs_ = np.empty(N)
        v_abs_[0] = v_abs_init

        ### add propagators to local scope

        for k, v in result["propagator"].items():
            if debug:
                print(" * Adding propagator: " + k + " = " + str(v.replace("__h", "h")))
            exec(k + " = " + v.replace("__h", "h"))

        for step in range(1, N):

            print("Step " + str(step) + " of " + str(N))

            ### set spike stimulus if necessary

            if step - 1 == spike_time_idx:
                for shape_idx in range(N_shapes):
                    shape_name = result["shape_state_variables"][shape_idx][0]
                    for shape_variable_idx, (shape_variable_name, shape_initial_value, shape_state_update) in enumerate(zip(result["shape_state_variables"], result["shape_initial_values"], result["shape_state_updates"])):
                        if shape_variable_name == "I_shape_ex__d":
                            shape_state[shape_name][shape_variable_name][step - 1] = i_ex_init[1]
                            break

            ### update V_abs

            V_abs_update_expr_ = V_abs_update_expr.copy()
            V_abs_update_expr_ = V_abs_update_expr_.subs(V_abs, v_abs_[step - 1])
            V_abs_update_expr_ = V_abs_update_expr_.subs(I_e, 0.)
            V_abs_update_expr_ = V_abs_update_expr_.subs(I_shape_ex, 0.)
            v_abs_[step] = V_abs_update_expr_

            ### update the state of each shape using propagators

            for shape_idx in range(N_shapes):
                shape_name = result["shape_state_variables"][shape_idx][0]
                
                if debug:
                    print("--> shape_name = " + shape_name)

                for shape_variable_idx, (shape_variable_name, shape_initial_value, shape_state_update) in enumerate(zip(result["shape_state_variables"][shape_idx], result["shape_initial_values"][shape_idx], result["shape_state_updates"][shape_idx])):
                    if debug:
                        print("\t* shape_variable_name = " + shape_variable_name)
                        print("\t* update expression = " + shape_state_update)
                    expr = eval(shape_state_update)
                    for _shape_name, _shape in shape_state.items():
                        for _var_name, _var_val in _shape.items():
                            expr = expr.subs(_var_name, _var_val[step - 1])
                    expr = expr.subs(Tau_syn_in, tau_syn)
                    expr = expr.subs(Tau_syn_ex, tau_syn)
                    if debug:
                        print("\t* update expression evaluates to = " + str(expr))
                    shape_state[shape_name][shape_variable_name][step] = expr

        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(1E3 * t, v_abs, label="V_abs (ref)")
        ax[0].plot(1E3 * t, v_abs_, linestyle=":", marker="o", label="V_abs (prop)")
        ax[0].set_ylabel("mV")
        ax[1].plot(1E3 * t, i_ex[0, :], label="i_ex (ref)")
        ax[1].plot(1E3 * t, shape_state["I_shape_ex"]["I_shape_ex"], linestyle=":", marker="o", label="i_ex (prop)")
        ax[2].plot(1E3 * t, i_ex[1, :], label="i_ex' (ref)")
        ax[2].plot(1E3 * t, shape_state["I_shape_ex"]["I_shape_ex__d"], linestyle=":", marker="o", label="i_ex' (prop)")
        for _ax in ax:
            _ax.legend()
        ax[0].set_xlabel("Time [ms]")
        plt.show()
        


        self.assertTrue(False)




if __name__ == '__main__':
    unittest.main()
