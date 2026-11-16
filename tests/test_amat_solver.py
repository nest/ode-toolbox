#
# test_amat_solver.py
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
import pytest
from scipy.integrate import odeint
import sympy

import odetoolbox

from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.debug_utils import import_matplotlib
from odetoolbox.spike_generator import SpikeGenerator


mpl, plt = import_matplotlib()
ENABLE_PLOTS: bool = mpl is not None


def neuron_ode(y, t, params):
    """
    Differential function for scipy.integrate.odeint based on Yamauchi et al. (2011).

    Parameters:
    -----------
    y : list or numpy.ndarray
        Current state vector [V_m, V_th_alpha_1, V_th_alpha_2, V_th_v_aux, V_th_v, I_syn_ex, I_syn_in]
    t : float
        Current time point
    params : dict
        Model parameters dictionary

    Returns:
    --------
    dydt : list
        Derivatives of the state variables [dV_m, dV_th_alpha_1, dV_th_alpha_2, dV_th_v_aux, dV_th_v, dI_syn_ex, dI_syn_in]
    """
    # Unpack state variables
    V_m, V_th_alpha_1, V_th_alpha_2, V_th_v_aux, V_th_v, I_syn_ex, I_syn_in = y

    # Unpack parameters
    C_m = params["C_m"]
    E_L = params["E_L"]
    I_e = params["I_e"]
    beta = params["beta"]
    tau_1 = params["tau_1"]
    tau_2 = params["tau_2"]
    tau_m = params["tau_m"]
    tau_syn_exc = params["tau_syn_exc"]
    tau_syn_inh = params["tau_syn_inh"]
    tau_v = params["tau_v"]

    # External stimulus current at time t
    I_stim = 0.0

    # Total synaptic current
    I_syn = I_syn_ex + I_syn_in

    # Membrane potential derivative (dV_dt)
    dV_dt = -(V_m - E_L) / tau_m + (I_syn + I_e + I_stim) / C_m

    # Threshold dynamics derivatives
    dV_th_alpha_1_dt = -V_th_alpha_1 / tau_1
    dV_th_alpha_2_dt = -V_th_alpha_2 / tau_2
    dV_th_v_aux_dt = -V_th_v_aux / tau_v + beta * dV_dt
    dV_th_v_dt = -V_th_v / tau_v + V_th_v_aux

    # Synaptic current convolutions translated to ODE decay form:
    # dI/dt = -I / tau_syn (Spike arrivals act as instantaneous jumps added externally)
    dI_syn_ex_dt = -I_syn_ex / tau_syn_exc
    dI_syn_in_dt = -I_syn_in / tau_syn_inh

    return [
        dV_dt,
        dV_th_alpha_1_dt,
        dV_th_alpha_2_dt,
        dV_th_v_aux_dt,
        dV_th_v_dt,
        dI_syn_ex_dt,
        dI_syn_in_dt
    ]


class TestAMATSolver:
    r"""Test propagators for AMAT neuron model. Make sure ODE-toolbox returns an analytic solver, and check the results by simulating the same model numerically via scipy."""

    def test_amat_solver(self):

        w: float = 1

        parameters = {
            "C_m": 200,
            "E_L": -70,
            "I_e": 0,
            "alpha_1": 10,
            "alpha_2": 0,
            "beta": 1,
            "omega": -65,
            "refr_T": 2,
            "tau_1": 10,
            "tau_2": 200,
            "tau_m": 10,
            "tau_syn_exc": 1,
            "tau_syn_inh": 3,
            "tau_v": 5
        }

        indict = {
            "dynamics": [
                {
                    "expression": "V_th_alpha_1' = -V_th_alpha_1 / tau_1",
                    "initial_values": {
                        "V_th_alpha_1": "0"
                    }
                },
                {
                    "expression": "V_th_alpha_2' = -V_th_alpha_2 / tau_2",
                    "initial_values": {
                        "V_th_alpha_2": "0"
                    }
                },
                {
                    "expression": "V_m' = ((-(V_m - E_L)) / tau_m + ((((unit_psc * I_kernel_exc__X__exc_spikes)) + ((-unit_psc * I_kernel_inh__X__inh_spikes))) + I_e + I_stim) / C_m)",
                    "initial_values": {
                        "V_m": str(parameters["E_L"])
                    }
                },
                {
                    "expression": "V_th_v_aux' = (-V_th_v_aux) / tau_v + beta * ((-(V_m - E_L)) / tau_m + ((((unit_psc * I_kernel_exc__X__exc_spikes)) + (((-unit_psc) * I_kernel_inh__X__inh_spikes))) + I_e + I_stim) / C_m)",
                    "initial_values": {
                        "V_th_v_aux": "0"
                    }
                },
                {
                    "expression": "V_th_v' = (-V_th_v) / tau_v + V_th_v_aux",
                    "initial_values": {
                        "V_th_v": "0"
                    }
                },
                {
                    "expression": "refr_t' = (-1)",
                    "initial_values": {
                        "refr_t": "0"
                    }
                },
                {
                    "expression": "I_kernel_exc__X__exc_spikes = exp(-t / tau_syn_exc)",
                    "initial_values": {}
                },
                {
                    "expression": "I_kernel_inh__X__inh_spikes = exp(-t / tau_syn_inh)",
                    "initial_values": {}
                }
            ],
            "parameters": {
                "C_m": str(parameters["C_m"]),
                "E_L": str(parameters["E_L"]),
                "I_e": str(parameters["I_e"]),
                "alpha_1": str(parameters["alpha_1"]),
                "alpha_2": str(parameters["alpha_2"]),
                "beta": str(parameters["beta"]),
                "omega": str(parameters["omega"]),
                "refr_T": str(parameters["refr_T"]),
                "tau_1": str(parameters["tau_1"]),
                "tau_2": str(parameters["tau_2"]),
                "tau_m": str(parameters["tau_m"]),
                "tau_syn_exc": str(parameters["tau_syn_exc"]),
                "tau_syn_inh": str(parameters["tau_syn_inh"]),
                "tau_v": str(parameters["tau_v"]),
                "unit_psc": "1",
                "I_stim": "0"
            }
        }
        dt = .1                               # time resolution (ms)
        T = 500.                              # simulation time (ms)
        input_spike_times = np.array([100., 300.])  # array of input spike times (ms)

        stimuli = [{"type": "list",
                    "list": " ".join([str(el) for el in input_spike_times]),
                    "variables": ["I_kernel_exc__X__exc_spikes"]}]

        spike_times = SpikeGenerator.spike_times_from_json(stimuli, T)

        # analyse with ODE-toolbox
        solver_dict = odetoolbox.analysis(indict, log_level="DEBUG", disable_stiffness_check=True)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        # simulate with ode-toolbox
        ODE_INITIAL_VALUES = {}
        for key, string_value in solver_dict["initial_values"].items():
            ODE_INITIAL_VALUES[key] = float(string_value)
            ODE_INITIAL_VALUES["I_kernel_exc__X__exc_spikes"] = 0.
            ODE_INITIAL_VALUES["I_kernel_inh__X__inh_spikes"] = 0.

        N = int(np.ceil(T / dt) + 1)
        timevec = np.linspace(0., T, N)
        analytic_integrator = AnalyticIntegrator(solver_dict, spike_times)
        analytic_integrator.shape_starting_values[sympy.Symbol("I_kernel_exc__X__exc_spikes", real=True)] = w
        analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)
        analytic_integrator.reset()
        state = {"timevec": []}

        for var_name in solver_dict["state_variables"]:
            state[var_name] = []

        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)
            state["timevec"].append(t)
            for sym, val in state_.items():
                state[str(sym)].append(val)

        # simulate with scipy odeint
        ts0 = np.arange(0., input_spike_times[0] - dt / 2, dt)
        ts1 = np.arange(input_spike_times[0], input_spike_times[1] - dt / 2, dt)
        ts2 = np.arange(input_spike_times[1], T + dt, dt)

        y0 = [parameters["E_L"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        w_one_hot_vec = [0, 0, 0, 0, 0, 1, 0]    # variable to which weight increments will be applied
        y_ = odeint(neuron_ode, y0, ts0, args=(parameters,))
        y_ = np.vstack([y_, odeint(neuron_ode, y_[-1, :] + w * w_one_hot_vec, ts1, args=(parameters,))])
        y_ = np.vstack([y_, odeint(neuron_ode, y_[-1, :] + w * w_one_hot_vec, ts2, args=(parameters,))])
        timevec_scipy = np.hstack([ts0, ts1, ts2])

        if ENABLE_PLOTS:
            fig, ax = plt.subplots(nrows=5, figsize=(5, 4), dpi=300)
            ax[0].plot(timevec, state["I_kernel_exc__X__exc_spikes"], "-", lw=2, color="k", label=r"$I_\mathrm{syn}$ (ODEtb)")
            ax[0].plot(timevec_scipy, y_[:, 5], "--", lw=2, color="orange", label=r"$I_\mathrm{syn}$ (scipy)", alpha=.5)

            ax[1].plot(timevec, state["V_th_v"], "-", lw=2, color="k", label=r"V_th_v (ODEtb)")
            ax[1].plot(timevec_scipy, y_[:, 4], "--", lw=2, color="orange", label=r"V_th_v (scipy)", alpha=.5)

            ax[2].plot(timevec, state["V_th_v_aux"], "-", lw=2, color="k", label=r"V_th_v_aux (ODEtb)")
            ax[2].plot(timevec_scipy, y_[:, 3], "--", lw=2, color="orange", label=r"V_th_v_aux (scipy)", alpha=.5)

            ax[3].plot(timevec, state["V_th_alpha_1"], "-", lw=2, color="k", label=r"V_th_alpha_1 (ODEtb)")
            ax[3].plot(timevec_scipy, y_[:, 1], "--", lw=2, color="orange", label=r"V_th_alpha_1 (scipy)", alpha=.5)

            ax[4].plot(timevec, state["V_th_alpha_2"], "-", lw=2, color="k", label=r"V_th_alpha_2 (ODEtb)")
            ax[4].plot(timevec_scipy, y_[:, 2], "--", lw=2, color="orange", label=r"V_th_alpha_2 (scipy)", alpha=.5)

            for _ax in ax:
                _ax.set_xlim(0., T + dt)
                _ax.legend()

            ax[-1].set_xlabel(r"time (ms)")

            fig.savefig("test_amat_solver.png")

        scipy_V_th_v_interp = np.interp(timevec_scipy, timevec, state["V_th_v"])
        scipy_V_th_v_aux_interp = np.interp(timevec_scipy, timevec, state["V_th_v_aux"])
        scipy_V_th_alpha_1_interp = np.interp(timevec_scipy, timevec, state["V_th_alpha_1"])
        scipy_V_th_alpha_2_interp = np.interp(timevec_scipy, timevec, state["V_th_alpha_2"])

        np.testing.assert_allclose(y_[:, 4], scipy_V_th_v_interp, atol=1E-7)
        np.testing.assert_allclose(y_[:, 3], scipy_V_th_v_aux_interp, atol=1E-7)
        np.testing.assert_allclose(y_[:, 1], scipy_V_th_alpha_1_interp, atol=1E-7)
        np.testing.assert_allclose(y_[:, 2], scipy_V_th_alpha_2_interp, atol=1E-7)
