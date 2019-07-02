#
# test_analytic_solver_integration.py
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
#np.seterr(under="warn")

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except:
    INTEGRATION_TEST_DEBUG_PLOTS = False


from .context import odetoolbox
from odetoolbox.analytic_integrator import AnalyticIntegrator

from math import e
from sympy import exp, sympify

import scipy
import scipy.special
import scipy.linalg
from scipy.integrate import solve_ivp

try:
    import pygsl.odeiv as odeiv
except ImportError as ie:
    print("Warning: PyGSL is not available. Test will be skipped.")
    print("Warning: " + str(ie))


def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestAnalyticSolverIntegration(unittest.TestCase):
    '''Numerical comparison between ode-toolbox calculated propagators, hand-calculated propagators expressed in Python, and numerical integration, for the iaf_cond_alpha neuron.

    The function tested is the alpha-shaped postsynaptic kernel.

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

          = [ (h/tau_syn + 1) * np.exp(-h/tau_syn)      h*np.exp(-h/tau_syn)                ]
            [ -h*np.exp(-h/tau_syn)/tau_syn**2          (-h/tau_syn + 1)*np.exp(-h/tau_syn) ]

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

    def test_analytic_solver_integration_psc_alpha(self):
        debug = True

        h = 1E-3    # [s]
        T = 20E-3    # [s]

        # neuron parameters
        tau = 20E-3    # [s]
        tau_syn = 5E-3    # [s]
        c_m = 1E-6    # [F]
        v_abs_init = -1000.   # [mV]
        i_ex_init = [0., e / tau_syn]   # [A]
        spike_times = [10E-3] # [s]
        for spike_time in spike_times:
            assert spike_time < T, "spike needs to occur before end of simulation"

        N = int(np.ceil(T / h) + 1)
        timevec = np.linspace(0., T, N)
        spike_times_idx = [np.argmin((spike_time - timevec)**2) for spike_time in spike_times]



        #
        #   compute numerical reference timeseries
        #

        def f(t, y):
            #print("In f(t=" + str(t) + ", y=" + str(y) + ")")
            i_ex = y[0:2]
            V_abs = y[2]

            _d_i_ex = np.array([i_ex[1], -i_ex[0] / tau_syn**2 - 2 * i_ex[1] / tau_syn])
            _d_V_abs_expr_ = -V_abs / tau + 3 * i_ex[0] / c_m    # factor 3 here because only simulating one inhibitory conductance, but ode-toolbox will add both inhibitory and excitatory and gap currents (which are of the exact same shape/magnitude at all times)
            _delta_vec = np.concatenate((_d_i_ex, [_d_V_abs_expr_]))
            #print("\treturning " + str(_delta_vec))

            return _delta_vec

        numerical_timevec = np.zeros((1, 0), dtype=np.float)
        numerical_sol = np.zeros((3, 0), dtype=np.float)

        _init_value = [0., 0., v_abs_init]

        numerical_timevec = np.hstack((numerical_timevec, np.array([0])[np.newaxis, :]))
        numerical_sol = np.hstack((numerical_sol, np.array(_init_value)[:, np.newaxis]))

        t = 0.
        spike_time_idx = 0
        while t < T:
            if spike_time_idx < len(spike_times):
                _t_stop = spike_times[spike_time_idx]
            else:
                _t_stop = T

            sol = solve_ivp(f, t_span=[t, _t_stop], y0=_init_value, t_eval=timevec[np.logical_and(t < timevec, timevec <= _t_stop)], rtol=1E-9, atol=1E-9)
            _init_value[:2] = i_ex_init       # "apply" the spike
            _init_value[2] = sol.y[2, -1]

            numerical_timevec = np.hstack((numerical_timevec, sol.t.copy()[np.newaxis, :]))
            numerical_sol = np.hstack((numerical_sol, sol.y.copy()))

            if numerical_timevec[0, -1] == spike_times[min(len(spike_times) - 1, spike_time_idx)]:
                numerical_sol[:2, -1] = i_ex_init   # set value to "just after" application of the spike, just for the log

            t = _t_stop
            spike_time_idx += 1

        i_ex = numerical_sol[:2, :]
        v_abs = numerical_sol[2, :]


        #
        #   timeseries using hand-calculated propagators (only for alpha postsynaptic currents, not V_abs)
        #

        P = scipy.linalg.expm(h * np.array([[0., 1.], [-1/tau_syn**2, -2/tau_syn]]))

        P_ = np.array([[ (h/tau_syn + 1) * np.exp(-h/tau_syn),  h*np.exp(-h/tau_syn)], \
                       [ -h*np.exp(-h/tau_syn)/tau_syn**2,   (-h/tau_syn + 1)*np.exp(-h/tau_syn) ]])

        np.testing.assert_allclose(P, P_)

        print("Propagator matrix from python-ref: " + str(P))
        assert len(spike_times_idx) == 1
        spike_time_idx = spike_times_idx[0]
        i_ex__ = np.zeros((2, N))
        for step in range(1, N):
            if step - 1 == spike_time_idx:
                i_ex__[:, step - 1] = i_ex_init
            i_ex__[:, step] = np.dot(P, i_ex__[:, step - 1])


        #
        #   timeseries using ode-toolbox generated propagators
        #

        indict = open_json("test_integration.json")
        solver_dict = odetoolbox.analysis(indict)
        print("Got solver_dict from ode-toolbox: ")
        print(json.dumps(solver_dict,  indent=2))
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        ODE_INITIAL_VALUES = { "V_abs" : v_abs_init, "I_shape_ex" : 0., "I_shape_ex__d" : 0., "I_shape_in" : 0., "I_shape_in__d" : 0., "I_shape_gap1" : 0., "I_shape_gap2" : 0. }

        _parms = {"Tau" : tau,
                 "Tau_syn_in" : tau_syn,
                 "Tau_syn_ex" : tau_syn,
                 "Tau_syn_gap" : tau_syn,
                 "C_m" : c_m,
                 "I_e" : 0.,
                 "currents" : 0.,
                 "e" : sympy.exp(1) }

        if not "parameters" in solver_dict.keys():
            solver_dict["parameters"] = {}
        solver_dict["parameters"].update(_parms)

        # ...
        spike_times_ = { "I_shape_ex__d" : spike_times, "I_shape_in__d" : spike_times, "I_shape_gap2" : spike_times }
        analytic_integrator = AnalyticIntegrator(solver_dict, spike_times_)
        analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)

        state = { sym : [] for sym in solver_dict["state_variables"] }
        state["timevec"] = []
        for step, t in enumerate(timevec):
            print("Step " + str(step) + " of " + str(N))
            state_ = analytic_integrator.get_value(t)
            state["timevec"].append(t)
            for sym, val in state_.items():
                state[sym].append(val)

        for k, v in state.items():
            state[k] = np.array(v)

        if INTEGRATION_TEST_DEBUG_PLOTS:
            fig, ax = plt.subplots(3, sharex=True)
            ax[0].plot(1E3 * numerical_timevec.squeeze(), v_abs, label="V_abs (num)")
            ax[0].plot(1E3 * state["timevec"], state["V_abs"], linestyle=":", marker="+", label="V_abs (prop)")

            ax[1].plot(1E3 * numerical_timevec.squeeze(), i_ex[0, :], linewidth=2, label="i (num)")
            ax[1].plot(1E3 * state["timevec"], state["I_shape_ex"], linewidth=2, linestyle=":", marker="o", label="i_ex (prop)")
            ax[1].plot(1E3 * state["timevec"], state["I_shape_in"], linewidth=2, linestyle=":", marker="o", label="i_in (prop)")
            ax[1].plot(1E3 * state["timevec"], state["I_shape_gap1"], linewidth=2, linestyle=":", marker="o", label="i_gap (prop)")
            ax[1].plot(1E3 * timevec, i_ex__[0, :], linewidth=2, linestyle="-.", marker="x", label="i_ex (prop ref)")

            ax[2].plot(1E3 * numerical_timevec.squeeze(), i_ex[1, :], linewidth=2, label="i_ex' (num)")
            ax[2].plot(1E3 * state["timevec"], state["I_shape_ex__d"], linewidth=2, linestyle=":", marker="o", label="i_ex' (prop)")
            ax[2].plot(1E3 * state["timevec"], state["I_shape_in__d"], linewidth=2, linestyle=":", marker="o", label="i_in' (prop)")
            ax[2].plot(1E3 * state["timevec"], state["I_shape_gap2"], linewidth=2, linestyle=":", marker="o", label="i_gap' (prop)")
            ax[2].plot(1E3 * timevec, i_ex__[1, :], linewidth=2, linestyle="-.", marker="x", label="i_ex' (prop ref)")

            for _ax in ax:
                _ax.legend()
                _ax.grid(True)
                #_ax.set_xlim(49., 55.)

            ax[-1].set_xlabel("Time [ms]")

            #plt.show()
            fn = "/tmp/remotefs2/propagators.png"
            print("Saving to " + fn)
            plt.savefig(fn, dpi=600)

        # the two propagators should be very close...
        np.testing.assert_allclose(i_ex__[0, :], state["I_shape_ex"], atol=1E-9, rtol=1E-9)
        np.testing.assert_allclose(i_ex__[0, :], state["I_shape_in"], atol=1E-9, rtol=1E-9)
        np.testing.assert_allclose(i_ex__[0, :], state["I_shape_gap1"], atol=1E-9, rtol=1E-9)
        np.testing.assert_allclose(i_ex__[1, :], state["I_shape_ex__d"], atol=1E-9, rtol=1E-9)
        np.testing.assert_allclose(i_ex__[1, :], state["I_shape_in__d"], atol=1E-9, rtol=1E-9)
        np.testing.assert_allclose(i_ex__[1, :], state["I_shape_gap2"], atol=1E-9, rtol=1E-9)

        # the numerical value is compared with a bit more leniency... compare max-normalised timeseries with the given rel, abs tolerances
        _num_norm_atol = 1E-4
        _num_norm_rtol = 1E-4
        np.testing.assert_allclose(i_ex__[0, :] / np.amax(np.abs(i_ex__[0, :])), i_ex[0, :] / np.amax(np.abs(i_ex__[0, :])), atol=_num_norm_atol, rtol=_num_norm_rtol)

        np.testing.assert_allclose(i_ex__[1, :] / np.amax(np.abs(i_ex__[1, :])), i_ex[1, :] / np.amax(np.abs(i_ex__[1, :])), atol=_num_norm_atol, rtol=_num_norm_rtol)

        np.testing.assert_allclose(v_abs / np.amax(v_abs), state["V_abs"] / np.amax(v_abs), atol=_num_norm_atol, rtol=_num_norm_rtol)

if __name__ == '__main__':
    unittest.main()
