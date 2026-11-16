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
import itertools
import logging
import numpy as np
import scipy
import sympy
import pytest

from .context import odetoolbox
from odetoolbox.analytic_integrator import AnalyticIntegrator, _parse_condition
from odetoolbox.debug_utils import import_matplotlib
from odetoolbox.singularity_detection import SingularityDetection
from odetoolbox.spike_generator import SpikeGenerator
from odetoolbox.system_of_shapes import SystemOfShapes
from odetoolbox.sympy_helpers import SymmetricEq, _sympy_parse_real, sympy_globals


mpl, plt = import_matplotlib()
ENABLE_PLOTS: bool = mpl is not None


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

    def test_undecidable_set_membership(self):
        r"""Test the case where sympy cannot decide whether the singular value is real, for instance :math:`1/g` (which is not real for :math:`g = 0`). The same condition is found once for each symbol (:math:`g = 1/\tau` and :math:`\tau = 1/g`); only one of them should remain after removing duplicates."""
        g, tau = sympy.symbols("g, tau", real=True)
        expr = 1 / (g * tau - 1)
        conditions = SingularityDetection._find_singularity_conditions_in_expression(expr)
        assert conditions == {SymmetricEq(g, 1 / tau), SymmetricEq(tau, 1 / g)}

        conditions = SingularityDetection._remove_duplicate_conditions(conditions)
        assert conditions == {SymmetricEq(g, 1 / tau)}

    def test_contradictory_conditions(self):
        r"""Test detection of combinations of conditions that cannot hold simultaneously."""
        g, k, tau = sympy.symbols("g, k, tau", real=True)

        # contradictory
        assert SingularityDetection._conditions_are_contradictory({SymmetricEq(g, 1 / tau), SymmetricEq(g, 0)})
        assert SingularityDetection._conditions_are_contradictory({SymmetricEq(k, g), SymmetricEq(k, 0), sympy.Ne(g, 0)})
        assert SingularityDetection._conditions_are_contradictory({SymmetricEq(k, 0), SymmetricEq(g, 0), sympy.Ne(k, g)})

        # not contradictory
        assert not SingularityDetection._conditions_are_contradictory({sympy.Ne(g, 0), sympy.Ne(k, g)})
        assert not SingularityDetection._conditions_are_contradictory({SymmetricEq(g, 1 / tau), sympy.Ne(g, 0)})
        assert not SingularityDetection._conditions_are_contradictory({SymmetricEq(k, g), SymmetricEq(k, 0), SymmetricEq(g, 0)})
        assert not SingularityDetection._conditions_are_contradictory({SymmetricEq(k, g), sympy.Ne(k, 0), sympy.Ne(g, 0)})

        # not contradictory (holds for all tau < 0), but ``sympy.solve()`` only finds the solution c = tau = 0
        c = sympy.Symbol("c", real=True)
        assert not SingularityDetection._conditions_are_contradictory({SymmetricEq(tau, -sympy.sqrt(c)), SymmetricEq(c, tau**2), sympy.Ne(tau, sympy.sqrt(c))})

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

        if ENABLE_PLOTS:

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

        np.testing.assert_allclose(actual, correct)
        np.testing.assert_allclose(y_[:, 1], state["I"], atol=1E-7)
        np.testing.assert_allclose(y_[:, 0], state["I_aux"], atol=1E-7)


class TestSingularityInDownstreamInhomogeneousTerm:
    r"""
    Test detection of singularities in the inhomogeneous update term :math:`Qb` of a row that has no inhomogeneous term of its own, but is driven by a row that does:

    .. math::

       y' = -k y + I
       x' = -g x + y

    The update term for :math:`x` contains a factor :math:`1 / g`, while :math:`b_x = 0`, so that the condition :math:`g = 0` cannot be found by only inspecting the particular solutions :math:`-b_i / A_{ii}`.

    The test is repeated with :math:`k = 1 / \tau`, so that rates and time constants are mixed. The propagator matrix then contains a factor :math:`1 / (g \tau - 1)`, whose singular value :math:`g = 1 / \tau` cannot be decided by sympy to be real.

    The value :math:`g = 0.5` coincides with :math:`k = 1 / \tau`, so that each of the conditional solvers is tested.
    """

    @pytest.mark.parametrize("g", [0., .3, .5])
    @pytest.mark.parametrize("parametrisation", ["rate", "time_constant"])
    def test_singularity_in_downstream_inhomogeneous_term(self, g: float, parametrisation: str):
        k = .5
        I = 2.
        x0 = -1.
        y0 = 1.

        if parametrisation == "rate":
            y_expr = "y' = -k * y + I"
            parameters = {"k": k, "g": g, "I": I}
        else:
            y_expr = "y' = -y / tau + I"
            parameters = {"tau": 1 / k, "g": g, "I": I}

        indict = {"dynamics": [{"expression": y_expr,
                                "initial_value": str(y0)},
                               {"expression": "x' = -g * x + y",
                                "initial_value": str(x0)}],
                  "options": {"output_timestep_symbol": "__h"}}

        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"
        assert any("g == 0" in condition for condition in solver_dict["conditions"].keys())
        if parametrisation == "time_constant":
            assert any("g == 1/tau" in condition for condition in solver_dict["conditions"].keys())

        solver_dict["parameters"] = parameters

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": x0, "y": y0})
        analytic_integrator.reset()

        timevec = np.linspace(0., 20., 161)
        actual = np.array([[analytic_integrator.get_value(t)[sympy.Symbol(sym, real=True)] for sym in ["y", "x"]] for t in timevec])

        def flow(state, t):
            y, x = state
            return [-k * y + I, -g * x + y]

        correct = scipy.integrate.odeint(flow, [y0, x0], timevec, rtol=1E-12, atol=1E-12)

        np.testing.assert_allclose(actual, correct, rtol=1E-7, atol=1E-7)


class TestSingularityCases:
    r"""
    For a range of dynamical systems, test that the expected singularity conditions are found, that no contradictory combinations of conditions are generated, and that the analytic solver is correct at singular as well as at generic parameter values (by comparison with ``scipy.integrate.odeint``).
    """

    # name: (dynamics as (state, right-hand side, initial value), expected equality conditions, parameter values to integrate at)
    CASES = {
        # conductance form: the leak can be switched off; C_m = 0 should not yield a condition
        "conductance_leak": ([("V", "(-g_L * (V - E_L) + I_e) / C_m", -70.)],
                             ["g_L == 0"],
                             [dict(g_L=.1, E_L=-70., I_e=5., C_m=2.), dict(g_L=0., E_L=-70., I_e=5., C_m=2.)]),
        # product of parameters
        "product_rate": ([("x", "-a * b * x + I", 1.)],
                         ["a == 0", "b == 0"],
                         [dict(a=.5, b=2., I=1.), dict(a=0., b=2., I=1.), dict(a=.5, b=0., I=1.), dict(a=0., b=0., I=1.)]),
        # shifted rate
        "shifted_rate": ([("x", "-(g - g_0) * x + I", 1.)],
                         ["g == g_0"],
                         [dict(g=.5, g_0=.2, I=1.), dict(g=.2, g_0=.2, I=1.)]),
        # singular values +/- sqrt(c) cannot be decided to be real
        "quadratic_rate": ([("x", "-(tau**2 - c) * x + I", 1.)],
                           ["c == tau**2", "tau == sqrt(c)", "tau == -sqrt(c)"],
                           [dict(tau=1., c=.5, I=1.), dict(tau=2., c=4., I=1.), dict(tau=-2., c=4., I=1.)]),
        # scaled time constants
        "scaled_tau": ([("y", "-y / tau_1", 1.), ("x", "y - 2 * x / tau_2", 0.)],
                       ["tau_1 == tau_2/2"],
                       [dict(tau_1=1., tau_2=3.), dict(tau_1=1., tau_2=2.)]),
        # repeated eigenvalue of the alpha kernel (Jordan block) coinciding with the membrane time constant
        "alpha_to_membrane": ([("I_aux", "-I_aux / tau_s", 1.), ("I", "I_aux - I / tau_s", 0.), ("V", "-V / tau_m + I / C", 0.)],
                              ["tau_s == tau_m"],
                              [dict(tau_s=2., tau_m=10., C=1.), dict(tau_s=2., tau_m=2., C=1.)]),
        # every combination of equal time constants in a chain of three
        "chain3_tau": ([("a", "-a / tau_1", 1.), ("b", "a - b / tau_2", 0.), ("c", "b - c / tau_3", 0.)],
                       ["tau_1 == tau_2", "tau_2 == tau_3", "tau_1 == tau_3"],
                       [dict(tau_1=1., tau_2=2., tau_3=3.), dict(tau_1=2., tau_2=2., tau_3=3.), dict(tau_1=1., tau_2=3., tau_3=3.),
                        dict(tau_1=3., tau_2=2., tau_3=3.), dict(tau_1=2., tau_2=2., tau_3=2.)]),
        # chain of three with rates and a constant input: zero rates (singular A) and equal rates
        "chain3_rate_inhom": ([("y", "-k * y + I", 1.), ("x", "-g * x + y", 0.), ("z", "-m * z + x", 0.)],
                              ["k == 0", "g == 0", "m == 0", "k == g", "g == m", "k == m"],
                              [dict(k=.5, g=.3, m=.2, I=1.), dict(k=0., g=.3, m=.2, I=1.), dict(k=.5, g=0., m=.2, I=1.),
                               dict(k=.5, g=.3, m=0., I=1.), dict(k=.5, g=.5, m=.5, I=1.), dict(k=0., g=0., m=0., I=1.),
                               dict(k=.5, g=0., m=.5, I=1.)]),
        # damped harmonic oscillator: critical damping is a repeated eigenvalue (see also ``test_damped_oscillator_generic``)
        "damped_oscillator": ([("x", "v", 1.), ("v", "-omega**2 * x - 2 * zeta * omega * v", 0.)],
                              ["zeta == 1", "zeta == -1", "omega == 0"],
                              [dict(omega=1., zeta=1.)]),
    }

    @staticmethod
    def _parse_condition_keys(conditions):
        r"""Return the set of equalities, and the list of AND-ed (in)equalities for each OR-ed term, from the condition keys of a solver dictionary."""
        and_terms = [set(and_term) for condition in conditions if condition != "default" for and_term in _parse_condition(condition)]
        equalities = {relation for and_term in and_terms for relation in and_term if isinstance(relation, SymmetricEq)}

        return equalities, and_terms

    @staticmethod
    def _integrate_and_compare(solver_dict, dynamics, parameters):
        state_syms = [sympy.Symbol(sym, real=True) for sym, _, _ in dynamics]
        x0 = [iv for _, _, iv in dynamics]
        timevec = np.linspace(0., 10., 41)

        param_syms = {sympy.Symbol(k, real=True): v for k, v in parameters.items()}
        local_dict = {str(sym): sym for sym in list(state_syms) + list(param_syms.keys())}
        rhs = [_sympy_parse_real(expr, global_dict=sympy_globals, local_dict=local_dict).subs(param_syms) for _, expr, _ in dynamics]
        f = sympy.lambdify(state_syms, rhs)
        correct = scipy.integrate.odeint(lambda y, t: f(*y), x0, timevec, rtol=1E-12, atol=1E-12)

        solver_dict["parameters"] = dict(parameters)
        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({str(sym): val for sym, val in zip(state_syms, x0)})
        analytic_integrator.reset()
        actual = np.array([[float(analytic_integrator.get_value(t)[sym]) for sym in state_syms] for t in timevec])

        np.testing.assert_allclose(actual, correct, rtol=1E-7, atol=1E-7, err_msg="for parameters " + str(parameters))

    @pytest.mark.parametrize("case", list(CASES.keys()))
    def test_singularity_cases(self, case: str):
        dynamics, expected_equalities, parameter_points = self.CASES[case]

        indict = {"dynamics": [{"expression": sym + "' = " + rhs, "initial_value": str(iv)} for sym, rhs, iv in dynamics],
                  "options": {"output_timestep_symbol": "__h"}}
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True)
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        # the expected conditions, and no others, are found
        equalities, and_terms = self._parse_condition_keys(solver_dict["conditions"].keys())
        assert equalities == {SymmetricEq(*[_sympy_parse_real(part.strip(), global_dict=sympy_globals) for part in eq.split("==")]) for eq in expected_equalities}

        # no solver is generated for contradictory combinations of conditions
        for cond_set in and_terms:
            assert not SingularityDetection._conditions_are_contradictory(cond_set), "contradictory condition: " + str(cond_set)

        # the solver is correct at each of the parameter points
        for parameters in parameter_points:
            self._integrate_and_compare(solver_dict, dynamics, parameters)

    @pytest.mark.xfail(strict=True, reason="for underdamped and overdamped oscillators, the propagators contain re(), im() and atan2(), which cannot be compiled")
    @pytest.mark.parametrize("zeta", [.5, 2.])
    def test_damped_oscillator_generic(self, zeta: float):
        dynamics, _, _ = self.CASES["damped_oscillator"]
        indict = {"dynamics": [{"expression": sym + "' = " + rhs, "initial_value": str(iv)} for sym, rhs, iv in dynamics],
                  "options": {"output_timestep_symbol": "__h"}}
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True)[0]
        self._integrate_and_compare(solver_dict, dynamics, dict(omega=1., zeta=zeta))

    @pytest.mark.parametrize("seed", range(4))
    def test_substitute_equalities_order(self, seed: int):
        r"""Substituting the equalities of a condition should give the same result regardless of their order (which, for a set, depends on the hash seed)."""
        k, g, m = sympy.symbols("k, g, m", real=True)
        M = sympy.Matrix([[-k, 0, 0], [1, -g, 0], [0, 1, -m]])
        equalities = [SymmetricEq(k, 0), SymmetricEq(g, k), SymmetricEq(m, 0), SymmetricEq(g, 0), SymmetricEq(g, m), SymmetricEq(k, m)]
        np.random.default_rng(seed).shuffle(equalities)
        assert SystemOfShapes._substitute_equalities(M, equalities) == sympy.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    @pytest.mark.parametrize("equal_pairs, num_distinct", [
        # all four time constants equal
        ([("t2", "t1"), ("t4", "t3"), ("t2", "t4"), ("t1", "t4"), ("t2", "t3"), ("t3", "t1")], 1),
        # three equal, one distinct
        ([("t1", "t2"), ("t2", "t3"), ("t3", "t1")], 2),
        # two pairs
        ([("t1", "t4"), ("t2", "t3")], 2)])
    def test_substitute_equalities_chain(self, equal_pairs, num_distinct: int):
        r"""For a chain of equalities between symbols, substituting one by one (even when repeated until nothing changes) can leave symbols in the result that should be equal, so that the conditional solver still contains the singularity it was supposed to remove. Test all orderings of the equalities."""
        t = {name: sympy.Symbol(name, real=True) for name in ["t1", "t2", "t3", "t4"]}
        M = sympy.Matrix([[-1 / t["t1"], 0, 0, 0], [1, -1 / t["t2"], 0, 0], [0, 1, -1 / t["t3"], 0], [0, 0, 1, -1 / t["t4"]]])
        equalities = [SymmetricEq(t[a], t[b]) for a, b in equal_pairs]

        # parameter values for which the equalities hold: one value per class of equal symbols
        classes = [{name} for name in t.keys()]
        for a, b in equal_pairs:
            class_a = next(cls for cls in classes if a in cls)
            class_b = next(cls for cls in classes if b in cls)
            if class_a is not class_b:
                classes.remove(class_b)
                class_a |= class_b

        values = {t[name]: sympy.Integer(i + 2) for i, cls in enumerate(classes) for name in cls}

        for perm in itertools.permutations(equalities):
            M_sub = SystemOfShapes._substitute_equalities(M, list(perm))
            assert len(M_sub.free_symbols) == num_distinct, "for order " + str(perm)
            assert M_sub.subs(values) == M.subs(values), "for order " + str(perm)

    def test_substitute_equalities_not_a_substitution(self):
        r"""Equalities that cannot be brought into the form of a substitution are skipped; the remaining ones are still substituted."""
        tau, c, g = sympy.symbols("tau, c, g", real=True)

        # :math:`\tau = -\sqrt{c}` becomes :math:`\tau = -\sqrt{\tau^2}` after substituting :math:`c = \tau^2`
        M = sympy.Matrix([[-(tau**2 - c)]])
        for equalities in [[SymmetricEq(c, tau**2), SymmetricEq(tau, -sympy.sqrt(c))], [SymmetricEq(tau, -sympy.sqrt(c)), SymmetricEq(c, tau**2)]]:
            assert SystemOfShapes._substitute_equalities(M, equalities) == sympy.Matrix([[0]])

        # :math:`g = 1/\tau \wedge g = \tau` has two discrete solutions (:math:`g = \tau = \pm 1`); only :math:`g = 1/\tau` is substituted
        M = sympy.Matrix([[-g, 0], [1, -1 / tau]])
        assert SystemOfShapes._substitute_equalities(M, [SymmetricEq(g, 1 / tau), SymmetricEq(g, tau)]) == sympy.Matrix([[-1 / tau, 0], [1, -1 / tau]])


#: condition string as generated for :math:`x' = -abc \cdot x + I` (see :py:meth:`TestConditionParsing.test_parse_condition_generated`): at least one of :math:`a`, :math:`b`, :math:`c` is zero
NESTED_CONDITION = "(((c == 0) && (b == 0) && (a == 0)) || (((b!=0) && (c!=0) && (a == 0)) || ((b == 0) && (c!=0) && (a!=0)))) || ((((b == 0) && (c!=0) && (a == 0)) || ((c == 0) && (b!=0) && (a!=0))) || (((c == 0) && (b!=0) && (a == 0)) || ((c == 0) && (b == 0) && (a!=0))))"


class TestConditionParsing:
    r"""Test parsing and evaluation of condition strings by the analytic integrator."""

    @pytest.mark.parametrize("condition, expected", [
        # single (in)equalities, with parentheses inside the expressions
        ("(tau == sqrt(c))", [["tau == sqrt(c)"]]),
        ("(tau!=-sqrt(c))", [["tau != -sqrt(c)"]]),
        ("((sqrt(c) + 1) == (tau))", [["sqrt(c) + 1 == tau"]]),
        ("((g == 1/tau))", [["g == 1/tau"]]),
        # a conjunction, as generated for each combination of conditions
        ("(a == 0) && (b!=sqrt(c)) && (c == (a + 1)**2)", [["a == 0", "b != sqrt(c)", "c == (a + 1)**2"]]),
        # nested disjunctions, as generated by repeatedly merging conditions
        ("((a == 0) && (b!=0)) || ((a!=0) && (b == 0))", [["a == 0", "b != 0"], ["a != 0", "b == 0"]]),
        ("(((a == 0) && (b!=sqrt(c))) || ((a!=0) && (b == 0))) || ((a == b) && (b == (c - 1)))", [["a == 0", "b != sqrt(c)"], ["a != 0", "b == 0"], ["a == b", "b == c - 1"]])])
    def test_parse_condition(self, condition: str, expected):
        def parse_relation(s):
            op = "==" if "==" in s else "!="
            lhs, rhs = [_sympy_parse_real(part, global_dict=sympy_globals) for part in s.split(op)]
            return SymmetricEq(lhs, rhs) if op == "==" else sympy.Ne(lhs, rhs)

        assert _parse_condition(condition) == [[parse_relation(s) for s in and_term] for and_term in expected]

    def test_parse_condition_generated(self):
        r"""For :math:`x' = -abc \cdot x + I`, the solvers for all combinations of the conditions :math:`a = 0`, :math:`b = 0` and :math:`c = 0` (except the default) are identical, and are merged into a single condition string with nested parentheses. Test that it is parsed into exactly these seven combinations."""
        indict = {"dynamics": [{"expression": "x' = -a * b * c * x + I", "initial_value": "1"}]}
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True)[0]
        conditions = [condition for condition in solver_dict["conditions"].keys() if condition != "default"]
        assert len(conditions) == 1

        a, b, c = sympy.symbols("a, b, c", real=True)
        combinations = set()
        for and_term in _parse_condition(conditions[0]):
            # each term specifies, for each of the parameters, whether it is zero or not
            assert len(and_term) == 3
            assert {rel.lhs for rel in and_term} == {a, b, c} and all(rel.rhs == 0 for rel in and_term)
            combinations.add(frozenset(rel.lhs for rel in and_term if isinstance(rel, SymmetricEq)))

        assert combinations == {frozenset(s) for n in [1, 2, 3] for s in itertools.combinations([a, b, c], n)}

    @pytest.mark.parametrize("condition, parameters, expected", [("(tau == sqrt(c))", dict(tau=2., c=4.), True),
                                                                 ("(tau == -sqrt(c))", dict(tau=2., c=4.), False),
                                                                 ("(tau!=sqrt(c)) && (c == tau**2)", dict(tau=-2., c=4.), True),
                                                                 ("(tau_1 == tau_2/2)", dict(tau_1=1., tau_2=2.), True),
                                                                 ("(tau == 1/g)", dict(tau=1., g=0.), False),
                                                                 ("(tau!=1/g)", dict(tau=1., g=0.), True),
                                                                 ("((a == 0) && (b == 0)) || ((a == 0) && (b!=0))", dict(a=0., b=1.), True),
                                                                 ("((a == 0) && (b == 0)) || ((a == 0) && (b!=0))", dict(a=1., b=0.), False),
                                                                 (NESTED_CONDITION, dict(a=1., b=0., c=1.), True),
                                                                 (NESTED_CONDITION, dict(a=0., b=0., c=0.), True),
                                                                 (NESTED_CONDITION, dict(a=1., b=2., c=3.), False)])
    def test_condition_holds(self, condition: str, parameters, expected: bool):
        integrator = AnalyticIntegrator.__new__(AnalyticIntegrator)
        integrator.solver_dict = {"parameters": parameters}
        assert integrator._condition_holds(condition) == expected


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
