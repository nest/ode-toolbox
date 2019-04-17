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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .context import odetoolbox
from odetoolbox.analytic import Propagator
from odetoolbox.shapes import shape_from_function

from math import e
from sympy import exp, sympify

from scipy.integrate import solve_ivp


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
        print("Got result from ode-toolbox: " + str(result))

        assert result["solver"] == "analytical"
        
        N_shapes = len(result["shape_state_variables"])
        

        h = 2.5E-4    # [s]
        T = 51E-3    # [s]

        # neuron parameters
        tau = 20E-3    # [s]
        tau_syn = 5E-3    # [s]
        c_m = 1E-6    # [F]
        v_abs_init = -70.   # [mV]
        i_ex_init = [0., e / tau_syn]   # [A]
        spike_time = 50E-3 # [s]
        assert spike_time < T, "spike needs to occur before end of simulation"

        N = int(np.ceil(T / h) + 1)
        t = np.linspace(0., T, N)
        spike_time_idx = np.argmin((spike_time - t)**2)


        #
        #   numerical reference timeseries
        #

        v_abs = np.empty(N)
        v_abs[0] = v_abs_init
        #I_shape_ex_expr_idx = np.where([indict["shapes"][i]["symbol"] == "I_shape_ex" for i in range(len(indict["shapes"]))])[0][0]
        #I_shape_ex_expr = eval(indict["shapes"][I_shape_ex_expr_idx]["definition"])
        #I_shape_ex = shape_from_function(indict["shapes"][I_shape_ex_expr_idx]["symbol"], indict["shapes"][I_shape_ex_expr_idx]["definition"])
        
        def f(t, y):
            print("In f(t=" + str(t) + ", y=" + str(y) + ")")
            i_ex = y[0:2]
            V_abs = y[2]

            _d_i_ex = np.array([i_ex[1], -i_ex[0] / tau_syn**2 - 2 * i_ex[1] / tau_syn])
            _d_V_abs_expr_ = -1/tau*V_abs + 1/c_m*(i_ex[0])
            _delta_vec = np.concatenate((_d_i_ex, [_d_V_abs_expr_]))
            print("\treturning " + str(_delta_vec))

            return _delta_vec
        
        sol = solve_ivp(f, t_span=[0., spike_time], y0=[0., 0., v_abs_init], t_eval=t[np.logical_and(0. <= t, t <= spike_time)])

        _sol_t_first_part = sol.t.copy()
        _sol_y_first_part = sol.y.copy()

        sol.y[:2, -1] = i_ex_init       # "apply" the spike
        sol = solve_ivp(f, t_span=[spike_time, T], y0=sol.y[:, -1], t_eval=t[np.logical_and(spike_time <= t, t < T)])
        _sol_t_second_part = sol.t.copy()
        _sol_y_second_part = sol.y.copy()
        
        _sol_t = np.hstack((_sol_t_first_part, _sol_t_second_part))
        _sol_y = np.hstack((_sol_y_first_part, _sol_y_second_part))
        
        i_ex = _sol_y[:2, :]
        v_abs = _sol_y[2, :]
        
        
        """i_ex = np.zeros((2, N))
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

            _d_i_ex = np.array([i_ex[1, step - 1], -i_ex[0, step - 1] / tau_syn**2 - 2 * i_ex[1, step - 1] / tau_syn])
            i_ex[:, step] = np.array(i_ex[:, step - 1]) + h * _d_i_ex
"""


        #
        #   timeseries using hand-calculated propagators
        #

        """ 

        alpha function: second-order

            g'' = -g / tau^2 - 2*g' / tau

        Let z1 = g
            z2 = g'

        Then z1' = z2
             z2' = -z1 / tau^2 - 2*z2 / tau

        Or equivalently

            Z' = S * Z

        with

            S = [ 0         1      ]
                [ -1/tau^2  -2/tau ]

        Exact solution: let

            P = exp[h * S]

        Then

            z(t + h) = P * z(t)

        """

        import scipy
        import scipy.linalg
        P = scipy.linalg.expm(h * np.array([[0., 1.], [-1/tau_syn**2, -2/tau_syn]]))

        P_ = np.array([[ (h/tau_syn + 1) * np.exp(-h/tau_syn),  h*np.exp(-h/tau_syn)], \
                       [ -h*np.exp(-h/tau_syn)/tau_syn**2,   (-h/tau_syn + 1)*np.exp(-h/tau_syn) ]])

        np.testing.assert_allclose(P, P_)

        print("Propagator matrix from python-ref: " + str(P))
        #v_abs__ = np.empty(N)
        #v_abs__[0] = v_abs_init
        i_ex__ = np.zeros((2, N))
        for step in range(1, N):
            if step - 1 == spike_time_idx:
                i_ex__[:, step - 1] = i_ex_init
            i_ex__[:, step] = np.dot(P, i_ex__[:, step - 1])




        #
        #   timeseries using ode-toolbox generated propagators
        #

        SHAPE_NAME_IDX = -1      # hard-coded index of shape variable name index; assumes that the order is [dy^n/dt^n, ..., dy/dn, y]
        ODE_INITIAL_VALUES = { "V_abs" : v_abs_init }

        ### define the necessary sympy variables (for shapes)

        Tau, Tau_syn_in, Tau_syn_ex, C_m,  I_e, currents = sympy.symbols("Tau Tau_syn_in Tau_syn_ex C_m  I_e currents")
        for shape_idx in range(N_shapes):
            for shape_variable_idx, (shape_variable_name, shape_initial_value, shape_state_update) in enumerate(zip(result["shape_state_variables"][shape_idx], result["shape_initial_values"][shape_idx], result["shape_state_updates"][shape_idx])):
                print(" * Defining var: " + str(shape_variable_name + " = sympy.symbols(\"" + shape_variable_name + "\")"))
                exec(shape_variable_name + " = sympy.symbols(\"" + shape_variable_name + "\")", globals())

        N_shape_variables = np.sum(len(shape_state_variable_names) for shape_state_variable_names in result["shape_state_variables"])


        ### define the necessary sympy variables (for ODE)

        ode_dim = len(result["ode_updates"])
        ode_state = {}
        ode_initial_values = {}
        ode_update_expressions = {}
        for ode_variable_name in result["ode_updates"].keys():
            print(" * Defining ODE variable: " + str(ode_variable_name))
            ode_state[ode_variable_name] = np.zeros(N)
            ode_state[ode_variable_name][0] = ODE_INITIAL_VALUES[ode_variable_name]
            ode_initial_values[ode_variable_name] = ODE_INITIAL_VALUES[ode_variable_name]
            ode_update_expressions[ode_variable_name] = result["ode_updates"][ode_variable_name]["constant"]
            print("   Update expression: " + str(ode_update_expressions[ode_variable_name]))
            exec(ode_variable_name + " = sympy.symbols(\"" + ode_variable_name + "\")", globals())


        ### define the necessary shape state variables

        N_shape_variables = np.sum(len(shape_state_variable_names) for shape_state_variable_names in result["shape_state_variables"])

        shape_state = {}
        shape_initial_values = {}
        for shape_idx in range(N_shapes):
            shape_name = result["shape_state_variables"][shape_idx][SHAPE_NAME_IDX]
            print("* Defining shape state variables for shape \'" + shape_name + "\'")
            shape_state[shape_name] = {}
            shape_initial_values[shape_name] = {}
            dim = len(result["shape_state_variables"])
            print("\t  dim = " + str(dim))
            for shape_variable_idx, (shape_variable_name, shape_initial_value, shape_state_update) in enumerate(zip(result["shape_state_variables"][shape_idx], result["shape_initial_values"][shape_idx], result["shape_state_updates"][shape_idx])):
                print("\t\t* Variable " + str(shape_variable_name))
                print("\t\t  Update expression: " + shape_state_update)
                print("\t\t  Initial value = " + str(shape_initial_value))
                expr = sympify(eval(shape_initial_value.replace("__h", "h")))
                expr = expr.subs(Tau, tau)
                expr = expr.subs(C_m, c_m)
                expr = expr.subs(Tau_syn_in, tau_syn)
                expr = expr.subs(Tau_syn_ex, tau_syn)
                print("\t\t     = " + str(expr))

                shape_state[shape_name][shape_variable_name] = np.zeros(N)
                shape_initial_values[shape_name][shape_variable_name] = expr


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
                    shape_name = result["shape_state_variables"][shape_idx][SHAPE_NAME_IDX]
                    for shape_variable_idx, (shape_variable_name, shape_initial_value, shape_state_update) in enumerate(zip(result["shape_state_variables"][shape_idx], result["shape_initial_values"][shape_idx], result["shape_state_updates"][shape_idx])):
                        shape_state[shape_name][shape_variable_name][step - 1] = shape_initial_values[shape_name][shape_variable_name]


            ### update the state of each shape using propagators

            for shape_idx in range(N_shapes):
                shape_name = result["shape_state_variables"][shape_idx][SHAPE_NAME_IDX]

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


            ### update the ODE state using propagators

            for ode_variable_name, ode_variable_val, in ode_state.items():
                print("\t* updating ODE variable " + str(ode_variable_name))
                
                ode_update_expression = ode_update_expressions[ode_variable_name]
                print("\t  expr = " + str(ode_update_expression))
                
                expr = sympify(eval(ode_update_expression.replace("__h", "h")))
                expr = expr.subs(Tau, tau)
                expr = expr.subs(C_m, c_m)
                expr = expr.subs(I_e, 0.)
                expr = expr.subs(currents, 0.)
                expr = expr.subs(Tau_syn_in, tau_syn)
                expr = expr.subs(Tau_syn_ex, tau_syn)
                expr = expr.subs(V_abs, ode_state[ode_variable_name][step - 1])
                
                ode_state[ode_variable_name][step] = expr
                print("\t  expr evaluates to = " + str(expr))
            
            """V_abs_update_expr_ = V_abs_update_expr.copy()
            V_abs_update_expr_ = V_abs_update_expr_.subs(V_abs, v_abs_[step - 1])
            V_abs_update_expr_ = V_abs_update_expr_.subs(I_e, shape_state["I_shape_ex"]["I_shape_ex"][step])
            V_abs_update_expr_ = V_abs_update_expr_.subs(I_shape_ex, 0.)
            v_abs_[step] = V_abs_update_expr_"""
            
            



        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(1E3 * _sol_t, v_abs, label="V_abs (num)")
        #ax[0].plot(1E3 * t, v_abs, label="V_abs (ref)")
        #ax[0].plot(1E3 * t, v_abs_, linestyle=":", marker="+", label="V_abs (prop)")
        ax[0].plot(1E3 * t, ode_state["V_abs"], linestyle=":", marker="+", label="V_abs (prop)")
        ax[0].set_ylabel("mV")

        #ax[1].plot(1E3 * t, i_ex[0, :], linewidth=2, label="i_ex (ref)")
        ax[1].plot(1E3 * _sol_t, i_ex[0, :], linewidth=2, label="i_ex (num)")
        ax[1].plot(1E3 * t, shape_state["I_shape_ex"]["I_shape_ex"], linewidth=2, linestyle=":", marker="o", label="i_ex (prop)")
        ax[1].plot(1E3 * t, i_ex__[0, :], linewidth=2, linestyle="-.", marker="x", label="i_ex (prop ref)")

        #ax[2].plot(1E3 * t, i_ex[1, :], linewidth=2, label="i_ex' (ref)")
        ax[2].plot(1E3 * _sol_t, i_ex[1, :], linewidth=2, label="i_ex' (num)")
        ax[2].plot(1E3 * t, shape_state["I_shape_ex"]["I_shape_ex__d"], linewidth=2, linestyle=":", marker="o", label="i_ex' (prop)")
        ax[2].plot(1E3 * t, i_ex__[1, :], linewidth=2, linestyle="-.", marker="x", label="i_ex (prop ref)")

        for _ax in ax:
            _ax.legend()
            _ax.grid(True)
            _ax.set_xlim(47., 53.)

        ax[-1].set_xlabel("Time [ms]")

        #plt.show()
        print("Saving to...")
        plt.savefig("/tmp/propagators.png", dpi=600)

        np.testing.assert_allclose(i_ex__[0, :], shape_state["I_shape_ex"]["I_shape_ex"])       # the two propagators should be very close
        np.testing.assert_allclose(i_ex__[0, :], i_ex[0, :], atol=1E-3, rtol=1E-3)        # the numerical value is compared with a bit more leniency
        
        np.testing.assert_allclose(i_ex[1, :], shape_state["I_shape_ex"]["I_shape_ex__d"])
        np.testing.assert_allclose(i_ex[1, :], i_ex__[1, :])

        np.testing.assert_allclose(v_abs, v_abs_)



if __name__ == '__main__':
    unittest.main()
