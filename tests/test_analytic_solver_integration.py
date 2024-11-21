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

import math
import numpy as np
import os
import sympy
import sympy.parsing.sympy_parser
import scipy
import scipy.special
import scipy.linalg
import scipy.integrate


try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except ImportError:
    INTEGRATION_TEST_DEBUG_PLOTS = False


from .context import odetoolbox
from odetoolbox.analytic_integrator import AnalyticIntegrator
from tests.test_utils import _open_json


class TestAnalyticSolverIntegration:
    r"""
    Numerical comparison between ode-toolbox calculated propagators, hand-calculated propagators expressed in Python, and numerical integration, for the iaf_cond_alpha neuron.

    The function tested is the alpha-shaped postsynaptic kernel.

    Definition of alpha function:

    .. math::

       g'' = -g / \tau^2 - 2 g' / \tau

    Let

    .. math::

       z_1 = g
       z_2 = g'

    Then

    .. math::

       z_1' = z_2
       z_2' = -z_1 / \tau^2 - 2 z_2 / \tau

    Or equivalently

    .. math::

        \mathbf{z}' = \mathbf{S} \mathbf{z}

    with

    .. math::

       \mathbf{S} = \left[\begin{matrix}0 & 1 \\ -1/\tau^2 & -2/\tau\end{matrix}\right]

    Exact solution: let

    .. math::

       \mathbf{P} &= \exp(h \cdot S)\\
                  &= \left[\begin{matrix}(h/\tau + 1) * \exp(-h/tau_syn) & h\cdot\exp(-h/\tau) \\
                                         -h \cdot \exp(-h/\tau)/\tau^2   & (-h/\tau + 1)\cdot\exp(-h/\tau)\end{matrix}\right]

    Then

    .. math::

        \mathbf{z}(t + h) = \mathbf{P} \cdot \mathbf{z}(t)
    """

    def test_analytic_solver_integration_psc_alpha(self):
        h = 1E-3    # [s]
        T = 20E-3    # [s]

        # neuron parameters
        tau = 20E-3    # [s]
        tau_syn = 5E-3    # [s]
        c_m = 1.    # [uF]
        I_ext = -1.  # [pA]
        V_rel_init = 0.   # [mV]
        i_ex_init = [0., math.e / tau_syn]   # [A]
        spike_times = [10E-3]  # [s]
        for spike_time in spike_times:
            assert spike_time < T, "spike needs to occur before end of simulation"

        N = int(np.ceil(T / h) + 1)
        timevec = np.linspace(0., T, N)
        spike_times_idx = [np.argmin((spike_time - timevec)**2) for spike_time in spike_times]

        #
        # compute numerical reference timeseries
        #

        def f(t, y):
            i_ex = y[:2]
            V_rel = y[2]

            _d_i_ex = np.array([i_ex[1], -i_ex[0] / tau_syn**2 - 2 * i_ex[1] / tau_syn])
            _d_V_rel_expr_ = -V_rel / tau + (3 * i_ex[0] + I_ext) / c_m    # factor 3 here because only simulating one inhibitory conductance, but ode-toolbox will add both inhibitory and excitatory and gap currents (which are of the exact same shape/magnitude at all times)
            _delta_vec = np.concatenate((_d_i_ex, [_d_V_rel_expr_]))

            return _delta_vec

        numerical_timevec = np.zeros((1, 0), dtype=float)
        numerical_sol = np.zeros((3, 0), dtype=float)

        _init_value = [0., 0., V_rel_init]

        numerical_timevec = np.hstack((numerical_timevec, np.array([0])[np.newaxis, :]))
        numerical_sol = np.hstack((numerical_sol, np.array(_init_value)[:, np.newaxis]))

        t = 0.
        spike_time_idx = 0
        while t < T:
            if spike_time_idx < len(spike_times):
                _t_stop = spike_times[spike_time_idx]
            else:
                _t_stop = T

            sol = scipy.integrate.solve_ivp(f, t_span=[t, _t_stop], y0=_init_value, t_eval=timevec[np.logical_and(t < timevec, timevec <= _t_stop)], rtol=1E-9, atol=1E-9)
            _init_value[:2] = i_ex_init       # "apply" the spike
            _init_value[2] = sol.y[2, -1]

            numerical_timevec = np.hstack((numerical_timevec, sol.t.copy()[np.newaxis, :]))
            numerical_sol = np.hstack((numerical_sol, sol.y.copy()))

            if numerical_timevec[0, -1] == spike_times[min(len(spike_times) - 1, spike_time_idx)]:
                numerical_sol[:2, -1] = i_ex_init   # set value to "just after" application of the spike, just for the log

            t = _t_stop
            spike_time_idx += 1

        i_ex = numerical_sol[:2, :]
        v_rel = numerical_sol[2, :]


        #
        #   timeseries using hand-calculated propagators (only for alpha postsynaptic currents, not V_rel)
        #

        P = scipy.linalg.expm(h * np.array([[0., 1.], [-1 / tau_syn**2, -2 / tau_syn]]))

        P_ = np.array([[(h / tau_syn + 1) * np.exp(-h / tau_syn), h * np.exp(-h / tau_syn)],
                       [-h * np.exp(-h / tau_syn) / tau_syn**2, (-h / tau_syn + 1) * np.exp(-h / tau_syn)]])

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

        print("Starting ODE-toolbox analysis...")
        indict = _open_json("test_integration.json")
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True, log_level="DEBUG")
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        ODE_INITIAL_VALUES = {"V_rel": V_rel_init, "I_shape_ex": 0., "I_shape_ex__d": 0., "I_shape_in": 0., "I_shape_in__d": 0., "I_shape_gap1": 0., "I_shape_gap2": 0.}

        _parms = {"Tau": tau,
                  "Tau_syn_in": tau_syn,
                  "Tau_syn_ex": tau_syn,
                  "Tau_syn_gap": tau_syn,
                  "C_m": c_m,
                  "I_ext": I_ext,
                  "e": sympy.exp(1)}

        if not "parameters" in solver_dict.keys():
            solver_dict["parameters"] = {}
        solver_dict["parameters"].update(_parms)

        spike_times_ = {"I_shape_ex__d": spike_times, "I_shape_in__d": spike_times, "I_shape_gap2": spike_times}
        analytic_integrator = AnalyticIntegrator(solver_dict, spike_times_)
        analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)

        state = {sym: [] for sym in solver_dict["state_variables"]}
        state["timevec"] = []
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)
            state["timevec"].append(t)
            for sym, val in state_.items():
                state[sym].append(val)

        for k, v in state.items():
            state[k] = np.array(v)

        if INTEGRATION_TEST_DEBUG_PLOTS:
            fig, ax = plt.subplots(3, sharex=True)
            ax[0].plot(1E3 * numerical_timevec.squeeze(), v_rel, label="V_rel (num)")
            ax[0].plot(1E3 * state["timevec"], state["V_rel"], linestyle=":", marker="+", label="V_rel (prop)")

            ax[1].plot(1E3 * numerical_timevec.squeeze(), i_ex[0, :], linewidth=2, label="i (num)")
            ax[1].plot(1E3 * state["timevec"], state["I_shape_ex"], linewidth=2, linestyle=":", marker="o", label="i_ex (prop)", fillstyle="none")
            ax[1].plot(1E3 * state["timevec"], state["I_shape_in"], linewidth=2, linestyle=":", marker="D", label="i_in (prop)", fillstyle="none")
            ax[1].plot(1E3 * state["timevec"], state["I_shape_gap1"], linewidth=2, linestyle=":", marker="^", label="i_gap (prop)", fillstyle="none")
            ax[1].plot(1E3 * timevec, i_ex__[0, :], linewidth=2, linestyle="-.", marker="x", label="i_ex (prop ref)")

            ax[2].plot(1E3 * numerical_timevec.squeeze(), i_ex[1, :], linewidth=2, label="i_ex' (num)")
            ax[2].plot(1E3 * state["timevec"], state["I_shape_ex__d"], linewidth=2, linestyle=":", marker="o", label="i_ex' (prop)", fillstyle="none")
            ax[2].plot(1E3 * state["timevec"], state["I_shape_in__d"], linewidth=2, linestyle=":", marker="D", label="i_in' (prop)", fillstyle="none")
            ax[2].plot(1E3 * state["timevec"], state["I_shape_gap2"], linewidth=2, linestyle=":", marker="^", label="i_gap' (prop)", fillstyle="none")
            ax[2].plot(1E3 * timevec, i_ex__[1, :], linewidth=2, linestyle="-.", marker="x", label="i_ex' (prop ref)")

            for _ax in ax:
                _ax.legend()
                _ax.grid(True)

            ax[-1].set_xlabel("Time [ms]")

            base_dir = "/tmp"
            fn = os.path.join(base_dir, "test_analytic_solver_integration.png")
            print("Saving to " + fn)
            plt.savefig(fn, dpi=600)
            plt.close(fig)

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

        np.testing.assert_allclose(v_rel / np.amax(v_rel), state["V_rel"] / np.amax(v_rel), atol=_num_norm_atol, rtol=_num_norm_rtol)
