#
# test_singularity_detection.py
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

import io
import logging
import numpy as np
import scipy
import sympy
import pytest

from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.spike_generator import SpikeGenerator

from .context import odetoolbox
from tests.test_utils import _open_json
from odetoolbox.singularity_detection import SingularityDetection
from odetoolbox.sympy_helpers import SymmetricEq, _sympy_parse_real

try:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except ImportError:
    INTEGRATION_TEST_DEBUG_PLOTS = False


class TestSingularityDetection:
    r"""Test singularity detection"""

    def test_is_matrix_defined_under_substitution(self):
        tau_m, tau_r, C, h = sympy.symbols("tau_m, tau_r, C, h", real=True)
        P = sympy.Matrix([[-1 / tau_r, 0, 0], [1, -1 / tau_r, 0], [0, 1 / C, -1 / tau_m]])
        assert SingularityDetection._is_matrix_defined_under_substitution(P, set())
        assert SingularityDetection._is_matrix_defined_under_substitution(P, set([SymmetricEq(tau_r, 1)]))
        assert not SingularityDetection._is_matrix_defined_under_substitution(P, set([SymmetricEq(tau_r, 0)]))

    @pytest.mark.parametrize("kernel_to_use", ["alpha", "beta"])
    def test_alpha_beta_kernels(self, kernel_to_use: str):
        r"""Test correctness of result for simple leaky integrate-and-fire neuron with biexponential postsynaptic kernel"""
        if kernel_to_use == "alpha":
            tau_m, tau_s, C, h = sympy.symbols("tau_m, tau_s, C, h", real=True)
            A = sympy.Matrix([[-1 / tau_s, 0, 0], [1, -1 / tau_s, 0], [0, 1 / C, -1 / tau_m]])
        elif kernel_to_use == "beta":
            tau_m, tau_d, tau_r, C, h = sympy.symbols("tau_m, tau_d, tau_r, C, h", real=True)
            A = sympy.Matrix([[-1 / tau_d, 0, 0], [1, -1 / tau_r, 0], [0, 1 / C, -1 / tau_m]])

        P = sympy.simplify(sympy.exp(A * h))  # Propagator matrix
        condition = SingularityDetection._generate_singularity_conditions(P)
        print(condition)
        condition = SingularityDetection._filter_valid_conditions(condition, A)  # filters out the invalid conditions (invalid means those for which A is not defined)

        if kernel_to_use == "alpha":
            assert len(condition) == 1
        elif kernel_to_use == "beta":
            assert len(condition) == 3

    def test_more_than_one_solution(self):
        r"""Test the case where there is more than one element returned in a solution to an equation; in this example, for a quadratic input equation"""
        tau_s = sympy.Symbol("tau_s", real=True)
        expr = _sympy_parse_real("-1/(tau_s**2 - 3*tau_s - 42)", local_dict={"tau_s": tau_s})
        A = sympy.Matrix([[expr]])
        conditions = SingularityDetection._generate_singularity_conditions(A)
        assert len(conditions) == 2
        for cond in conditions:
            assert sympy.Symbol("tau_s", real=True) == cond.lhs
            assert cond.rhs == _sympy_parse_real("3/2 + sqrt(177)/2") \
                or cond.rhs == _sympy_parse_real("3/2 - sqrt(177)/2")


class TestSingularityInBothPropagatorAndInhomogeneous:
    r"""
    Test singularity mitigations when there is simultaneously a potential singularity in the propagator matrix as well as in the inhomogeneous terms.
    """

    @pytest.mark.parametrize("tau_1, tau_2", [(10., 2.), (10., 10.)])
    @pytest.mark.parametrize("late_ltd_check, late_ltp_check", [(3.14, 2.71), (0., 0.)])
    def test_singularity_in_both_propagator_and_inhomogeneous(self, tau_1, tau_2, late_ltd_check, late_ltp_check):

        def time_to_max(tau_1, tau_2):
            r"""
            Time of maximum.
            """
            if tau_1 == tau_2:
                return tau_1

            return (np.log(tau_1) - np.log(tau_2)) / (1. / tau_2 - 1. / tau_1)

        def unit_amplitude(tau_1, tau_2):
            r"""
            Scaling factor ensuring that amplitude of solution is one.
            """
            tmax = time_to_max(tau_1, tau_2)

            return 1. / (np.exp(-tmax / tau_1) - np.exp(-tmax / tau_2))

        def double_exponential_ode_flow(y, t, tau_1, tau_2, alpha, dt):
            r"""
            Rhs of ODE system to be solved.
            """
            dy1dt = -y[0] / tau_1
            dy2dt = y[0] - y[1] / tau_2

            return np.array([dy1dt, dy2dt])

        def inhomogeneous_ode_flow(z, t, late_ltp_check, late_ltd_check, tau_z):
            """
            Defines the differential equation for z.
            dz/dt = f(z, t)
            """
            dzdt = (((1.0 - z) * late_ltp_check - (z + 0.5) * late_ltd_check)) / tau_z
            return dzdt

        dt = .125                             # time resolution (ms)
        T = 48.    # simulation time (ms)

        w = 3.14                              # weight (amplitude; pA)
        alpha = 1.
        input_spike_times = np.array([10., 32.])  # array of input spike times (ms)
        stimuli = [{"type": "list",
                    "list": " ".join([str(el) for el in input_spike_times]),
                    "variables": ["I_aux"]}]

        spike_times = SpikeGenerator.spike_times_from_json(stimuli, T)

        indict = {"dynamics": [{"expression": "I_aux' = -I_aux / tau_1",    # double exponential
                                "initial_values": {"I_aux": "0."}},
                               {"expression": "I' = I_aux - I / tau_2",    # double exponential
                                "initial_values": {"I": "0"}},
                               {"expression": "z' = (((1 - z) * late_ltp_check) - (z + 0.5) * late_ltd_check) / tau_z",
                                "initial_value": "1"}],    # ODE with inhomogeneous term
                  "options": {"output_timestep_symbol": "__h"},
                  "parameters": {"tau_1": str(tau_1),
                                 "tau_2": str(tau_2),
                                 "w": str(w),
                                 "alpha": str(alpha)}}

        #
        #    integration using the ODE-toolbox analytic integrator
        #

        timevec = np.arange(0., T, dt)

        solver_dict = odetoolbox.analysis(indict, log_level="DEBUG", disable_stiffness_check=True)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        # solver_dict["parameters"] = {}
        solver_dict["parameters"]["tau_z"] = 20.
        solver_dict["parameters"]["late_ltd_check"] = late_ltd_check
        solver_dict["parameters"]["late_ltp_check"] = late_ltp_check

        N = int(np.ceil(T / dt) + 1)
        timevec = np.linspace(0., T, N)
        analytic_integrator = AnalyticIntegrator(solver_dict, spike_times)
        analytic_integrator.shape_starting_values[sympy.Symbol("I_aux", real=True)] = w * alpha
        ODE_INITIAL_VALUES = {"I": 0., "I_aux": 0., "z": 0.}
        analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)
        analytic_integrator.reset()
        state = {"timevec": [], "I": [], "I_aux": [], "z": []}
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)
            state["timevec"].append(t)
            for sym, val in state_.items():
                state[str(sym)].append(val)

        actual = [analytic_integrator.get_value(t)[sympy.Symbol("z", real=True)] for t in timevec]

        #
        #    integration using scipy.integrate.odeint
        #

        z0 = 0.0      # set the initial condition
        params = solver_dict["parameters"]
        ode_args = (
            params["late_ltp_check"],
            params["late_ltd_check"],
            params["tau_z"]
        )

        solution = scipy.integrate.odeint(inhomogeneous_ode_flow, z0, timevec, args=ode_args, rtol=1E-12, atol=1E-12)
        correct = solution.flatten().tolist()

        ts0 = np.arange(0., input_spike_times[0] + dt / 2, dt)
        ts1 = np.arange(input_spike_times[0], input_spike_times[1] + dt / 2, dt)
        ts2 = np.arange(input_spike_times[1], T + dt, dt)

        y_ = scipy.integrate.odeint(double_exponential_ode_flow, [0., 0.], ts0, args=(tau_1, tau_2, alpha, dt), rtol=1E-12, atol=1E-12)
        y_ = np.vstack([y_[:-1, :], scipy.integrate.odeint(double_exponential_ode_flow, [y_[-1, 0] + w * alpha, y_[-1, 1]], ts1, args=(tau_1, tau_2, alpha, dt), rtol=1E-12, atol=1E-12)])
        y_ = np.vstack([y_[:-1, :], scipy.integrate.odeint(double_exponential_ode_flow, [y_[-1, 0] + w * alpha, y_[-1, 1]], ts2, args=(tau_1, tau_2, alpha, dt), rtol=1E-12, atol=1E-12)])

        ts0 = ts0[:-1]
        ts1 = ts1[:-1]

        if INTEGRATION_TEST_DEBUG_PLOTS:

            #
            #   plot the double exponential ODE
            #

            tmax = time_to_max(tau_1, tau_2)
            mpl.rcParams["text.usetex"] = True

            fig, ax = plt.subplots(nrows=6, figsize=(5, 4), dpi=600)
            ax[0].plot(timevec, state["I"], "-", lw=3, color="k", label=r"$I(t)$ (ODEtb)")
            ax[0].plot(np.hstack([ts0, ts1, ts2]), y_[:, 1], "-", lw=2, color="r", label=r"$I(t)$ (odeint)")
            ax[1].plot(timevec, state["I_aux"], "--", lw=3, color="k", label=r"$I_\mathsf{aux}(t)$ (ODEtb)")
            ax[1].plot(np.hstack([ts0, ts1, ts2]), y_[:, 0], "--", lw=2, color="r", label=r"$I_\mathsf{aux}(t)$ (odeint)")

            for tin in input_spike_times:
                ax[0].vlines(tin + tmax, ax[0].get_ylim()[0], ax[0].get_ylim()[1], colors="k", linestyles=":")
                ax[1].vlines(tin, ax[0].get_ylim()[0], ax[0].get_ylim()[1], colors="k", linestyles=":")

            ax[2].semilogy(np.hstack([ts0, ts1, ts2]), np.abs(y_[:, 1] - state["I"]), label="I")
            ax[2].semilogy(np.hstack([ts0, ts1, ts2]), np.abs(y_[:, 0] - state["I_aux"]), linestyle="--", label="I_aux")
            ax[2].set_ylabel("Error")

            ax[3].plot(timevec, correct, label="z (odeint)")
            ax[4].plot(timevec, actual, label="z (ODEtb)")
            ax[5].semilogy(timevec, np.abs(np.array(correct) - np.array(actual)), label="z")
            ax[5].set_ylabel("Error")
            ax[-1].set_xlabel("Time")
            for _ax in ax:
                _ax.set_xlim(0., T + dt)
                _ax.legend()
                _ax.grid()
                if not _ax == ax[-1]:
                    _ax.set_xticklabels([])

            fig.savefig("/tmp/test_singularity_simultaneous_[tau_1=" + str(tau_1) + "]_[tau_2=" + str(tau_2) + "]_[late_ltd_check=" + str(late_ltd_check) + "]_[late_ltp_check=" + str(late_ltp_check) + "].png")

        #
        #   test
        #

        np.testing.assert_allclose(correct, actual)
        np.testing.assert_allclose(y_[:, 1], state["I"], atol=1E-7)
        np.testing.assert_allclose(y_[:, 0], state["I_aux"], atol=1E-7)


class TestPropagatorSolverHomogeneous:
    r"""Test ODE-toolbox ability to ignore imaginary roots of the dynamical equations.

    This dynamical system is difficult for sympy to solve and it needs to do quite some number crunching.

    Test that no singularity conditions are found for this system after analysis.
    """

    def test_propagator_solver_homogeneous(self):
        indict = {"dynamics": [{"expression": "V_m' = (1.0E+03 * ((V_m * 1.0E+03) / ((tau_m * 1.0E+15) * (1 + exp(alpha_exp * (V_m_init * 1.0E+03))))))",
                                "initial_values": {"V_m": "(1.0E+03 * (-70.0 * 1.0E-03))"}}],
                  "options": {"output_timestep_symbol": "__h",
                              "simplify_expression": "sympy.logcombine(sympy.powsimp(sympy.expand(expr)))"},
                  "parameters": {"V_m_init": "(1.0E+03 * (-65.0 * 1.0E-03))",
                                 "alpha_exp": "2 / ((3.0 * 1.0E+06))",
                                 "tau_m": "(1.0E+15 * (2.0 * 1.0E-03))"}}

        logger = logging.getLogger()

        log_stream = io.StringIO()
        log_handler = logging.StreamHandler(log_stream)
        logger.addHandler(log_handler)

        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True, log_level="DEBUG")

        log_contents = log_stream.getvalue()

        # test that no singularity conditions were found for this system
        assert not "Under certain conditions" in log_contents
        assert not "division by zero" in log_contents
