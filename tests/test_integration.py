#
# test_integration.py
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

from math import e
from sympy import exp, sympify

import scipy
import scipy.linalg
from scipy.integrate import solve_ivp



def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestSolutionComputation(unittest.TestCase):
    '''Numerical comparison between ode-toolbox calculated propagators, hand-calculated propagators expressed in Python, and numerical integration, for the iaf_cond_alpha neuron.

    Definition of alpha function:

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
        print("Got result from ode-toolbox: ")
        print(json.dumps(result,  indent=2))
        assert result["solver"] == "analytical"

        shape_names = result["shape_state_updates"].keys()
        assert shape_names == result["shape_initial_values"].keys()
        N_shapes = len(shape_names)

        h = 1E-3    # [s]
        T = 100E-3    # [s]

        # neuron parameters
        tau = 20E-3    # [s]
        tau_syn = 5E-3    # [s]
        c_m = 1E-6    # [F]
        v_abs_init = -1000.   # [mV]
        i_ex_init = [0., e / tau_syn]   # [A]
        spike_time = 10E-3 # [s]
        assert spike_time < T, "spike needs to occur before end of simulation"

        N = int(np.ceil(T / h) + 1)
        t = np.linspace(0., T, N)
        spike_time_idx = np.argmin((spike_time - t)**2)


        #
        #   numerical reference timeseries
        #

        def f(t, y):
            print("In f(t=" + str(t) + ", y=" + str(y) + ")")
            i_ex = y[0:2]
            V_abs = y[2]

            _d_i_ex = np.array([i_ex[1], -i_ex[0] / tau_syn**2 - 2 * i_ex[1] / tau_syn])
            _d_V_abs_expr_ = -1/tau*V_abs + 1/c_m*(2*i_ex[0])    # XXX: factor 2 here because only simulating one inhibitory conductance, but ode-toolbox will add both inhibitory and excitatory currents (which are of the exact same shape/magnitude at all times)
            _delta_vec = np.concatenate((_d_i_ex, [_d_V_abs_expr_]))
            print("\treturning " + str(_delta_vec))

            return _delta_vec

        sol = solve_ivp(f, t_span=[0., spike_time], y0=[0., 0., v_abs_init], t_eval=t[np.logical_and(0. <= t, t <= spike_time)], rtol=1E-9, atol=1E-9)

        sol.y[:2, -1] = i_ex_init       # "apply" the spike

        _sol_t_first_part = sol.t.copy()
        _sol_y_first_part = sol.y.copy()

        sol = solve_ivp(f, t_span=[spike_time, T], y0=sol.y[:, -1], t_eval=t[np.logical_and(spike_time < t, t <= T)])
        _sol_t_second_part = sol.t.copy()
        _sol_y_second_part = sol.y.copy()

        _sol_t = np.hstack((_sol_t_first_part, _sol_t_second_part))
        _sol_y = np.hstack((_sol_y_first_part, _sol_y_second_part))

        i_ex = _sol_y[:2, :]
        v_abs = _sol_y[2, :]


        #
        #   timeseries using hand-calculated propagators
        #

        P = scipy.linalg.expm(h * np.array([[0., 1.], [-1/tau_syn**2, -2/tau_syn]]))

        P_ = np.array([[ (h/tau_syn + 1) * np.exp(-h/tau_syn),  h*np.exp(-h/tau_syn)], \
                       [ -h*np.exp(-h/tau_syn)/tau_syn**2,   (-h/tau_syn + 1)*np.exp(-h/tau_syn) ]])

        np.testing.assert_allclose(P, P_)

        print("Propagator matrix from python-ref: " + str(P))
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

        ### define the necessary sympy variables

        Tau, Tau_syn_in, Tau_syn_ex, C_m,  I_e, currents = sympy.symbols("Tau Tau_syn_in Tau_syn_ex C_m  I_e currents")


        ### define the necessary sympy variables (for shapes)

        for shape_name in shape_names:
            print("* Defining variables for shape " + str(shape_name))
            shape_variables = result["shape_initial_values"][shape_name].keys()
            assert set(shape_variables) == set(result["shape_state_updates"][shape_name].keys())
            for shape_variable in shape_variables:
                shape_variable_code_name = shape_variable.replace("'", "__d")
                shape_variable_initial_value = result["shape_initial_values"][shape_name][shape_variable]
                shape_variable_update_expr = result["shape_state_updates"][shape_name][shape_variable]
                print("\t * Defining var: " + str(shape_variable) + " as sympy.symbols(\"" + shape_variable_code_name + "\")")
                exec(shape_variable_code_name + " = sympy.symbols(\"" + shape_variable_code_name + "\")", globals())


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
            ode_update_expressions[ode_variable_name] = " + ".join(["(" + v + ")" for k, v in result["ode_updates"][ode_variable_name].items()])
            print("   Update expression: " + str(ode_update_expressions[ode_variable_name]))
            exec(ode_variable_name + " = sympy.symbols(\"" + ode_variable_name + "\")", globals())


        ### define the necessary shape state variables
        
        # state and initial values are stored here by their "code name", i.e. with "__d" instead of "'" for derivative; this is to make it proper Python syntax to e.g. put a derivative on the left-hand side of an assignment

        shape_state = {}
        shape_initial_values = {}
        for shape_name in shape_names:
            print("* Defining shape state variables for shape \'" + shape_name + "\'")
            shape_state[shape_name] = {}
            shape_initial_values[shape_name] = {}
            dim = len(result["shape_state_updates"])
            print("\t  shape dim = " + str(dim))

            shape_variables = result["shape_initial_values"][shape_name].keys()
            assert shape_variables == result["shape_state_updates"][shape_name].keys()
            for shape_variable in shape_variables:
                shape_variable_code_name = shape_variable.replace("'", "__d")
                shape_variable_initial_value = result["shape_initial_values"][shape_name][shape_variable]
                shape_variable_update_expr = result["shape_state_updates"][shape_name][shape_variable]

                print("\t\t* Variable " + str(shape_variable_code_name))
                print("\t\t  Update expression: " + shape_variable_update_expr)
                print("\t\t  Initial value = " + str(shape_variable_initial_value))
                expr = sympify(eval(shape_variable_initial_value.replace("__h", "h")))
                expr = expr.subs(Tau, tau)
                expr = expr.subs(C_m, c_m)
                expr = expr.subs(Tau_syn_in, tau_syn)
                expr = expr.subs(Tau_syn_ex, tau_syn)
                print("\t\t     = " + str(expr))

                shape_state[shape_name][shape_variable_code_name] = np.zeros(N)
                shape_initial_values[shape_name][shape_variable_code_name] = expr


        ### add propagators to local scope

        for k, v in result["propagator"].items():
            if debug:
                print(" * Adding propagator: " + k + " = " + str(v.replace("__h", "h")))
            exec(k + " = " + v.replace("__h", "h"))

        for step in range(1, N):

            print("Step " + str(step) + " of " + str(N))

            ### set spike stimulus if necessary

            if step - 1 == spike_time_idx:
                for shape_name in shape_names:
                    shape_variables = result["shape_initial_values"][shape_name].keys()
                    assert shape_variables == result["shape_state_updates"][shape_name].keys()
                    for shape_variable in shape_variables:
                        shape_variable_code_name = shape_variable.replace("'", "__d")
                        shape_variable_initial_value = result["shape_initial_values"][shape_name][shape_variable]
                        shape_variable_update_expr = result["shape_state_updates"][shape_name][shape_variable]
                        shape_state[shape_name][shape_variable_code_name][step - 1] = shape_initial_values[shape_name][shape_variable_code_name]


            ### update the state of each shape using propagators

            for shape_name in shape_names:
                if debug:
                    print("--> shape_name = " + shape_name)

                shape_variables = result["shape_initial_values"][shape_name].keys()
                for shape_variable in shape_variables:
                    shape_variable_code_name = shape_variable.replace("'", "__d")
                    shape_variable_initial_value = result["shape_initial_values"][shape_name][shape_variable]
                    shape_variable_update_expr = result["shape_state_updates"][shape_name][shape_variable]
                    if debug:
                        print("\t* shape_variable_name = " + shape_variable_code_name)
                        print("\t* update expression = " + shape_variable_update_expr)

                    # replace symbolic variables with their values at the last step
                    expr = eval(shape_variable_update_expr)
                    for _shape_name, _shape in shape_state.items():
                        for _var_name, _var_val in _shape.items():
                            _var_name_code_name = _var_name.replace("'", "__d")
                            expr = expr.subs(_var_name, shape_state[_shape_name][_var_name_code_name][step - 1])  # _var_val [step - 1])
                            print("\t\t* replacing variable " + _var_name + " with " + str(shape_state[_shape_name][_var_name_code_name][step - 1]))
                    expr = expr.subs(Tau_syn_in, tau_syn)
                    expr = expr.subs(Tau_syn_ex, tau_syn)
                    if debug:
                        print("\t* update expression evaluates to = " + str(expr))
                    shape_state[shape_name][shape_variable_code_name][step] = expr


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
                
                # replace all shape variable names by their values
                for _shape_name, _shape in shape_state.items():
                    for _var_name, _var_val in _shape.items():
                        expr = expr.subs(_var_name, _var_val[step - 1])
                
                ode_state[ode_variable_name][step] = expr
                print("\t  expr evaluates to = " + str(expr))

        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(1E3 * _sol_t, v_abs, label="V_abs (num)")
        ax[0].plot(1E3 * t, ode_state["V_abs"], linestyle=":", marker="+", label="V_abs (prop)")

        ax[1].plot(1E3 * _sol_t, i_ex[0, :], linewidth=2, label="i_ex (num)")
        ax[1].plot(1E3 * t, shape_state["I_shape_ex"]["I_shape_ex"], linewidth=2, linestyle=":", marker="o", label="i_ex (prop)")
        ax[1].plot(1E3 * t, i_ex__[0, :], linewidth=2, linestyle="-.", marker="x", label="i_ex (prop ref)")

        ax[2].plot(1E3 * _sol_t, i_ex[1, :], linewidth=2, label="i_ex' (num)")
        ax[2].plot(1E3 * t, shape_state["I_shape_ex"]["I_shape_ex__d"], linewidth=2, linestyle=":", marker="o", label="i_ex' (prop)")
        ax[2].plot(1E3 * t, i_ex__[1, :], linewidth=2, linestyle="-.", marker="x", label="i_ex (prop ref)")

        for _ax in ax:
            _ax.legend()
            _ax.grid(True)
            #_ax.set_xlim(49., 55.)

        ax[-1].set_xlabel("Time [ms]")

        #plt.show()
        print("Saving to...")
        plt.savefig("/tmp/remotefs2/propagators.png", dpi=600)

        # the two propagators should be very close...
        np.testing.assert_allclose(i_ex__[0, :], shape_state["I_shape_ex"]["I_shape_ex"], atol=1E-9, rtol=1E-9)
        #np.testing.assert_allclose(i_ex__[1, :], shape_state["I_shape_ex"]["I_shape_ex__d"], atol=1E-9, rtol=1E-9)    # XXX: cannot check this, as ode-toolbox conversion to lower triangular format changes the semantics/behaviour of I_shape_ex__d; see eq. 14 in Blundell et al. 2018 Front Neuroinformatics

        # the numerical value is compared with a bit more leniency... compare max-normalised timeseries with the given rel, abs tolerances
        _num_norm_atol = 1E-4
        _num_norm_rtol = 1E-4
        np.testing.assert_allclose(i_ex__[0, :] / np.amax(np.abs(i_ex__[0, :])), i_ex[0, :] / np.amax(np.abs(i_ex__[0, :])), atol=_num_norm_atol, rtol=_num_norm_rtol)

        np.testing.assert_allclose(i_ex__[1, :] / np.amax(np.abs(i_ex__[1, :])), i_ex[1, :] / np.amax(np.abs(i_ex__[1, :])), atol=_num_norm_atol, rtol=_num_norm_rtol)

        np.testing.assert_allclose(v_abs / np.amax(v_abs), ode_state["V_abs"] / np.amax(v_abs), atol=_num_norm_atol, rtol=_num_norm_rtol)

if __name__ == '__main__':
    unittest.main()
