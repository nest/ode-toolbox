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

import os
import tempfile

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

        w_exc: float = 100.    # increment of the excitatory synaptic current per spike (pA)
        w_inh: float = 50.     # increment of the inhibitory synaptic current per spike (pA)

        parameters = {
            "C_m": 200,
            "E_L": -70,
            "I_e": 150,    # nonzero, so that the inhomogeneous terms in V_m and V_th_v_aux are exercised
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
                        "V_th_alpha_1": "3"
                    }
                },
                {
                    "expression": "V_th_alpha_2' = -V_th_alpha_2 / tau_2",
                    "initial_values": {
                        "V_th_alpha_2": "1"
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
        dt = .1                                    # time resolution (ms)
        T = 500.                                   # simulation time (ms)
        exc_spike_times = np.array([100., 300.])   # excitatory input spike times (ms)
        inh_spike_times = np.array([200., 400.])   # inhibitory input spike times (ms)

        stimuli = [{"type": "list",
                    "list": " ".join([str(el) for el in exc_spike_times]),
                    "variables": ["I_kernel_exc__X__exc_spikes"]},
                   {"type": "list",
                    "list": " ".join([str(el) for el in inh_spike_times]),
                    "variables": ["I_kernel_inh__X__inh_spikes"]}]

        spike_times = SpikeGenerator.spike_times_from_json(stimuli, T)

        # analyse with ODE-toolbox
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True, disable_singularity_detection=True)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        # simulate with ode-toolbox; the synaptic kernels start at zero and are incremented at each spike
        ODE_INITIAL_VALUES = {key: float(string_value) for key, string_value in solver_dict["initial_values"].items()}
        ODE_INITIAL_VALUES["I_kernel_exc__X__exc_spikes"] = 0.
        ODE_INITIAL_VALUES["I_kernel_inh__X__inh_spikes"] = 0.

        N = int(np.ceil(T / dt) + 1)
        timevec = np.linspace(0., T, N)
        analytic_integrator = AnalyticIntegrator(solver_dict, spike_times)
        analytic_integrator.shape_starting_values[sympy.Symbol("I_kernel_exc__X__exc_spikes", real=True)] = w_exc
        analytic_integrator.shape_starting_values[sympy.Symbol("I_kernel_inh__X__inh_spikes", real=True)] = w_inh
        analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)
        analytic_integrator.reset()

        state = {var_name: [] for var_name in solver_dict["state_variables"]}
        for t in timevec:
            for sym, val in analytic_integrator.get_value(t).items():
                state[str(sym)].append(val)

        state = {var_name: np.array(values, dtype=float) for var_name, values in state.items()}

        # simulate with scipy odeint, restarting the integration at each spike with the state incremented by the synaptic weight. Note that the inhibitory kernel enters the ODE for ``V_m`` with a negative sign, while the scipy state variable ``I_syn_in`` enters with a positive sign; hence the inhibitory increment is negative.
        scipy_state_idx = {"V_m": 0, "V_th_alpha_1": 1, "V_th_alpha_2": 2, "V_th_v_aux": 3, "V_th_v": 4, "I_kernel_exc__X__exc_spikes": 5, "I_kernel_inh__X__inh_spikes": 6}
        scipy_state_sign = {var_name: -1. if var_name == "I_kernel_inh__X__inh_spikes" else 1. for var_name in scipy_state_idx.keys()}

        spike_events = [(t_sp, scipy_state_idx["I_kernel_exc__X__exc_spikes"], w_exc) for t_sp in exc_spike_times]
        spike_events.extend([(t_sp, scipy_state_idx["I_kernel_inh__X__inh_spikes"], -w_inh) for t_sp in inh_spike_times])
        spike_events.sort()
        segment_boundaries = [0.] + [t_sp for t_sp, _, _ in spike_events] + [T + dt]

        y_start = np.zeros(len(scipy_state_idx))
        for var_name, idx in scipy_state_idx.items():
            y_start[idx] = scipy_state_sign[var_name] * ODE_INITIAL_VALUES[var_name]

        y_segments = []
        timevec_segments = []
        for i in range(len(segment_boundaries) - 1):
            if i > 0:
                # apply the spike at the start of this segment
                _, spike_idx, spike_weight = spike_events[i - 1]
                y_start = y_end.copy()
                y_start[spike_idx] += spike_weight

            # integrate up to and including the next segment boundary, so that the state exactly at the next spike time is known; the boundary point itself is not part of the comparison
            ts = np.arange(segment_boundaries[i], segment_boundaries[i + 1] - dt / 2, dt)
            y = odeint(neuron_ode, y_start, np.append(ts, segment_boundaries[i + 1]), args=(parameters,), rtol=1E-12, atol=1E-12)
            y_end = y[-1, :]
            y_segments.append(y[:-1, :])
            timevec_segments.append(ts)

        y_ = np.vstack(y_segments)
        timevec_scipy = np.hstack(timevec_segments)

        if ENABLE_PLOTS:
            fig, ax = plt.subplots(nrows=len(scipy_state_idx), figsize=(5, 2 * len(scipy_state_idx)), dpi=150)
            for _ax, (var_name, idx) in zip(ax, scipy_state_idx.items()):
                _ax.plot(timevec, state[var_name], "-", lw=2, color="k", label=var_name + " (ODEtb)")
                _ax.plot(timevec_scipy, scipy_state_sign[var_name] * y_[:, idx], "--", lw=2, color="orange", label=var_name + " (scipy)", alpha=.5)
                _ax.set_xlim(0., T + dt)
                _ax.legend()

            ax[-1].set_xlabel("time (ms)")

            fig.savefig(os.path.join(tempfile.gettempdir(), "test_amat_solver.png"))
            plt.close(fig)

        # compare all state variables of the model
        for var_name, idx in scipy_state_idx.items():
            odetb_interp = np.interp(timevec_scipy, timevec, state[var_name])
            np.testing.assert_allclose(scipy_state_sign[var_name] * y_[:, idx], odetb_interp, rtol=1E-7, atol=1E-7, err_msg="for state variable " + var_name)

        # the refractoriness timer decreases linearly
        np.testing.assert_allclose(state["refr_t"], ODE_INITIAL_VALUES["refr_t"] - timevec, atol=1E-9)
