#
# test_double_exponential.py
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

import numpy as np
from scipy.integrate import odeint

import odetoolbox

from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.spike_generator import SpikeGenerator

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except ImportError:
    INTEGRATION_TEST_DEBUG_PLOTS = False


class TestDoubleExponential:
    r"""Test propagators generation for double exponential"""

    def test_double_exponential(self):
        r"""Test propagators generation for double exponential"""

        def time_to_max(tau_1, tau_2):
            r"""
            Time of maximum.
            """
            tmax = (np.log(tau_1) - np.log(tau_2)) / (1. / tau_2 - 1. / tau_1)
            return tmax

        def unit_amplitude(tau_1, tau_2):
            r"""
            Scaling factor ensuring that amplitude of solution is one.
            """
            tmax = time_to_max(tau_1, tau_2)
            alpha = 1. / (np.exp(-tmax / tau_1) - np.exp(-tmax / tau_2))
            return alpha

        def flow(y, t, tau_1, tau_2, alpha, dt):
            r"""
            Rhs of ODE system to be solved.
            """
            dy1dt = -y[0] / tau_1
            dy2dt = y[0] - y[1] / tau_2

            return np.array([dy1dt, dy2dt])

        indict = {"dynamics": [{"expression": "I_aux' = -I_aux / tau_1",
                                "initial_values": {"I_aux": "0."}},
                               {"expression": "I' = I_aux - I / tau_2",
                                "initial_values": {"I": "0"}}],
                  "options": {"output_timestep_symbol": "__h"},
                  "parameters": {"tau_1": "10",
                                 "tau_2": "2",
                                 "w": "3.14",
                                 "alpha": str(unit_amplitude(tau_1=10., tau_2=2.)),
                                 "weighted_input_spikes": "0."}}

        w = 3.14                              # weight (amplitude; pA)
        tau_1 = 10.                           # decay time constant (ms)
        tau_2 = 2.                            # rise time constant (ms)
        dt = .125                             # time resolution (ms)
        T = 500.                              # simulation time (ms)
        input_spike_times = np.array([100., 300.])  # array of input spike times (ms)

        alpha = unit_amplitude(tau_1, tau_2)

        stimuli = [{"type": "list",
                    "list": " ".join([str(el) for el in input_spike_times]),
                    "variables": ["I_aux"]}]

        spike_times = SpikeGenerator.spike_times_from_json(stimuli, T)

        ODE_INITIAL_VALUES = {"I": 0., "I_aux": 0.}

        # simulate with ode-toolbox
        solver_dict = odetoolbox.analysis(indict, log_level="DEBUG", disable_stiffness_check=True)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        N = int(np.ceil(T / dt) + 1)
        timevec = np.linspace(0., T, N)
        analytic_integrator = AnalyticIntegrator(solver_dict, spike_times)
        analytic_integrator.shape_starting_values["I_aux"] = w * alpha * (1. / tau_2 - 1. / tau_1)
        analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)
        analytic_integrator.reset()
        state = {"timevec": [], "I": [], "I_aux": []}
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)
            state["timevec"].append(t)
            for sym, val in state_.items():
                state[sym].append(val)

        # solve with odeint
        ts0 = np.arange(0., input_spike_times[0] - dt / 2, dt)
        ts1 = np.arange(input_spike_times[0], input_spike_times[1] - dt / 2, dt)
        ts2 = np.arange(input_spike_times[1], T + dt, dt)

        y_ = odeint(flow, [0., 0.], ts0, args=(tau_1, tau_2, alpha, dt))
        y_ = np.vstack([y_, odeint(flow, [y_[-1, 0] + w * alpha * (1. / tau_2 - 1. / tau_1), y_[-1, 1]], ts1, args=(tau_1, tau_2, alpha, dt))])
        y_ = np.vstack([y_, odeint(flow, [y_[-1, 0] + w * alpha * (1. / tau_2 - 1. / tau_1), y_[-1, 1]], ts2, args=(tau_1, tau_2, alpha, dt))])

        rec_I_interp = np.interp(np.hstack([ts0, ts1, ts2]), timevec, state['I'])
        rec_I_aux_interp = np.interp(np.hstack([ts0, ts1, ts2]), timevec, state['I_aux'])

        if INTEGRATION_TEST_DEBUG_PLOTS:
            tmax = time_to_max(tau_1, tau_2)
            mpl.rcParams['text.usetex'] = True

            fig, ax = plt.subplots(nrows=2, figsize=(5, 4), dpi=300)
            ax[0].plot(timevec, state['I_aux'], '--', lw=3, color='k', label=r'$I_\mathsf{aux}(t)$ (NEST)')
            ax[0].plot(timevec, state['I'], '-', lw=3, color='k', label=r'$I(t)$ (NEST)')
            ax[0].plot(np.hstack([ts0, ts1, ts2]), y_[:, 0], '--', lw=2, color='r', label=r'$I_\mathsf{aux}(t)$ (odeint)')
            ax[0].plot(np.hstack([ts0, ts1, ts2]), y_[:, 1], '-', lw=2, color='r', label=r'$I(t)$ (odeint)')

            for tin in input_spike_times:
                ax[0].vlines(tin + tmax, ax[0].get_ylim()[0], ax[0].get_ylim()[1], colors='k', linestyles=':')

            ax[1].semilogy(np.hstack([ts0, ts1, ts2]), np.abs(y_[:, 1] - rec_I_interp), label="I")
            ax[1].semilogy(np.hstack([ts0, ts1, ts2]), np.abs(y_[:, 0] - rec_I_aux_interp), linestyle="--", label="I_aux")
            ax[1].set_ylabel("Error")

            for _ax in ax:
                _ax.set_xlim(0., T + dt)
                _ax.legend()

            ax[-1].set_xlabel(r'time (ms)')

            fig.savefig('double_exp_test.png')

        np.testing.assert_allclose(y_[:, 1], rec_I_interp, atol=1E-7)

    def test_constant_factors_double_exponential(self):
        r"""Test the computation of propagators for an alpha (double-exponential) kernel with constant coefficients; this tests the block-diagonal computation of propagators."""
        indict = {"dynamics": [{"expression": "x'' = -2 * x' - x",
                                "initial_values": {"x": "0",
                                                   "x'": "0"}}]}
        solver_dict = odetoolbox.analysis(indict, log_level="DEBUG", disable_stiffness_check=True)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"
        assert len(solver_dict["propagators"]) == 4
