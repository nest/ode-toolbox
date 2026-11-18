#
# test_inhomogeneous.py
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

from typing import Callable, Dict, List

import numpy as np
import pytest
import scipy.integrate
import sympy

import odetoolbox

from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.shapes import Shape
from odetoolbox.sympy_helpers import SymmetricEq
from odetoolbox.system_of_shapes import SystemOfShapes


def _integrate_and_compare(solver_dict, flow: Callable[[List[float], float], List[float]], x0: Dict[str, float], parameters: Dict[str, float], t_max: float = 20.):
    r"""
    Integrate the analytic solver in ``solver_dict`` for the given parameter values, and compare the result to a numerical reference obtained with ``scipy.integrate.odeint``.

    ``flow(y, t)`` returns the derivatives of the state variables, in the order of the keys of ``x0``.
    """
    state_names = list(x0.keys())
    timevec = np.linspace(0., t_max, 81)

    analytic_integrator = AnalyticIntegrator(dict(solver_dict, parameters=dict(parameters)))
    analytic_integrator.set_initial_values(x0)
    analytic_integrator.reset()
    actual = np.array([[float(analytic_integrator.get_value(t)[sympy.Symbol(name, real=True)]) for name in state_names] for t in timevec])

    correct = scipy.integrate.odeint(flow, [x0[name] for name in state_names], timevec, rtol=1E-12, atol=1E-12)

    np.testing.assert_allclose(actual, correct, rtol=1E-7, atol=1E-7, err_msg="for parameters " + str(parameters))


def _analyse(dynamics) -> Dict:
    r"""Run ODE-toolbox on the given dynamics, and check that a single analytic solver is returned."""
    result = odetoolbox.analysis({"dynamics": dynamics}, disable_stiffness_check=True)
    assert len(result) == 1
    assert result[0]["solver"] == "analytical"

    return result[0]


class TestInhomogeneous:
    """Test correct propagators returned for simple inhomogeneous ODEs"""

    @pytest.mark.parametrize("dt", [.1, 1.])
    def test_constant_rate(self, dt: float):
        r"""Test an ODE of the form x' = 42 with x(t = 0) = -42."""
        x0 = -42.

        shape = Shape.from_ode("x", "42", initial_values={"x": str(x0)})
        sys_of_shape = SystemOfShapes.from_shapes([shape])
        solver_dict = sys_of_shape.generate_propagator_solver()

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": str(x0)})
        analytic_integrator.reset()

        actual = []
        correct = []
        cur_x = x0
        timevec = np.arange(0., 100., dt)
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)[sympy.Symbol("x", real=True)]
            actual.append(state_)

            cur_x = x0 + 42 * t
            correct.append(cur_x)

        np.testing.assert_allclose(actual, correct)

    @pytest.mark.parametrize("dt", [.1, 1.])
    @pytest.mark.parametrize("ode_definition", ["(U - x) / tau",
                                                "(1 - x) / tau"])
    def test_inhomogeneous_solver(self, dt, ode_definition):
        U = .2
        if ode_definition == "(1 - x) / tau":
            U = 1.
        tau = 5.  # [s]

        x0 = 0.

        parameters_dict = {sympy.Symbol("U", real=True): str(U),
                           sympy.Symbol("tau", real=True): str(tau)}

        shape = Shape.from_ode("x", ode_definition, initial_values={"x": str(x0)}, parameters=parameters_dict)

        assert not shape.is_homogeneous()
        assert shape.is_lin_const_coeff()

        sys_of_shape = SystemOfShapes.from_shapes([shape], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()
        solver_dict["parameters"] = {str(sym): expr for sym, expr in parameters_dict.items()}

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": str(x0)})
        analytic_integrator.reset()

        actual = []
        correct = []
        cur_x = x0
        timevec = np.arange(0., 100., dt)
        kernel = np.exp(-dt / tau)
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)[sympy.Symbol("x", real=True)]
            actual.append(state_)
            correct.append(cur_x)
            cur_x = U + kernel * (cur_x - U)

        np.testing.assert_allclose(actual, correct)

    @pytest.mark.parametrize("dt", [.1, 1.])
    def test_inhomogeneous_simultaneous(self, dt: float):
        U = .2
        tau = 5.  # [s]

        x0 = 0.

        parameters_dict = {sympy.Symbol("U", real=True): str(U),
                           sympy.Symbol("tau1", real=True): str(tau),
                           sympy.Symbol("tau2", real=True): str(tau)}

        shape_x = Shape.from_ode("x", "(U - x) / tau1", initial_values={"x": str(x0)}, parameters=parameters_dict)
        shape_y = Shape.from_ode("y", "(1 - y) / tau2", initial_values={"y": str(x0)}, parameters=parameters_dict)

        sys_of_shape = SystemOfShapes.from_shapes([shape_x, shape_y], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()
        solver_dict["parameters"] = {str(sym): expr for sym, expr in parameters_dict.items()}

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": str(x0), "y": str(x0)})
        analytic_integrator.reset()

        actual_x = []
        actual_y = []
        correct_x = []
        correct_y = []
        cur_x = x0
        cur_y = x0
        timevec = np.arange(0., 100., dt)
        kernel = np.exp(-dt / tau)
        for step, t in enumerate(timevec):
            state_x = analytic_integrator.get_value(t)[sympy.Symbol("x", real=True)]
            state_y = analytic_integrator.get_value(t)[sympy.Symbol("y", real=True)]
            actual_x.append(state_x)
            actual_y.append(state_y)
            correct_x.append(cur_x)
            correct_y.append(cur_y)
            cur_x = U + kernel * (cur_x - U)
            cur_y = 1 + kernel * (cur_y - 1)

        np.testing.assert_allclose(correct_x, actual_x)
        np.testing.assert_allclose(correct_y, actual_y)

    def test_inhomogeneous_solver_second_order(self):
        r"""Test propagator generation for an inhomogeneous second-order ODE, given as a single shape."""
        tau = 10.  # [s]
        parameters_dict = {sympy.Symbol("tau", real=True): str(tau)}

        x0 = 0.
        x0d = 10.

        shape = Shape.from_ode("x", "-1/tau**2 * x - 2/tau * x' - 1", initial_values={"x": str(x0), "x'": str(x0d)}, parameters=parameters_dict)
        sys_of_shape = SystemOfShapes.from_shapes([shape], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()

        _integrate_and_compare(solver_dict,
                               lambda y, t: [y[1], -y[0] / tau**2 - 2 * y[1] / tau - 1],
                               {"x": x0, "x__d": x0d},
                               {"tau": tau},
                               t_max=100.)

    def test_inhomogeneous_solver_second_order_system(self):
        r"""Test propagator generation for an inhomogeneous second-order ODE, given as a system of two first-order shapes."""
        tau = 10.  # [s]
        parameters_dict = {sympy.Symbol("tau", real=True): str(tau)}

        x0 = 0.
        x0d = 10.

        shape1 = Shape.from_ode("x", "y", initial_values={"x": str(x0)}, parameters=parameters_dict)
        shape2 = Shape.from_ode("y", "-1/tau**2 * x - 2/tau * y - 1", initial_values={"y": str(x0d)}, parameters=parameters_dict)
        sys_of_shape = SystemOfShapes.from_shapes([shape1, shape2], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()

        _integrate_and_compare(solver_dict,
                               lambda y, t: [y[1], -y[0] / tau**2 - 2 * y[1] / tau - 1],
                               {"x": x0, "y": x0d},
                               {"tau": tau},
                               t_max=100.)

    def test_inhomogeneous_solver_second_order_system_api(self):
        r"""Test that an analytic solver is returned for an inhomogeneous second-order ODE when called via analysis()"""
        indict = {"dynamics": [{"expression": "x' = y",
                                "initial_value": "0."},
                               {"expression": "y' = -1/tau**2 * x - 2/tau * y - 1",
                                "initial_value": "0."}],
                  "parameters": {"tau": "10."}}

        result = odetoolbox.analysis(indict, disable_stiffness_check=True)
        assert len(result) == 1 \
               and result[0]["solver"] == "analytical"

    def test_inhomogeneous_solver_second_order_combined_system(self):
        r"""test propagators generation for combined homogeneous/inhomogeneous ODEs"""
        tau = 10.  # [s]
        E_L = -70.  # [mV]
        parameters_dict = {sympy.Symbol("tau", real=True): str(tau),
                           sympy.Symbol("E_L", real=True): str(E_L)}

        x0 = 0.
        x0d = 10.

        shape_V_m = Shape.from_ode("V_m", "x / tau + (E_L - V_m)", initial_values={"V_m": "0."}, parameters=parameters_dict)
        shape_I_syn1 = Shape.from_ode("x", "y", initial_values={"x": str(x0)}, parameters=parameters_dict)
        shape_I_syn2 = Shape.from_ode("y", "-1/tau**2 * x - 2/tau * y", initial_values={"y": str(x0d)}, parameters=parameters_dict)
        sys_of_shape = SystemOfShapes.from_shapes([shape_V_m, shape_I_syn1, shape_I_syn2], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()
        assert set(solver_dict["state_variables"]) == set(['V_m', 'x', 'y'])

    def test_inhomogeneous_solver_second_order_combined_system_api(self):
        r"""test propagators generation for combined homogeneous/inhomogeneous ODEs when called via analysis()"""
        indict = {"dynamics": [{"expression": "x' = y",
                                "initial_value": "0."},
                               {"expression": "y' = -1/tau**2 * x - 2/tau * y",
                                "initial_value": "0."},
                               {"expression": "V_m' = x / tau + (E_L - V_m)",
                                "initial_value": "0"}],
                  "parameters": {"tau": "10.",
                                 "E_L": "-70."}}

        result = odetoolbox.analysis(indict)
        assert len(result) == 1 \
               and result[0]["solver"] == "analytical"

    def test_inhomogeneous_solver_combined_system(self):
        r"""test propagators generation for combined homogeneous/inhomogeneous ODEs when called via analysis()"""
        indict = {"dynamics": [{"expression": "x' = a * 0.001",
                                "initial_value": "0.3"},
                               {"expression": "y' = -y / b",
                                "initial_value": "0"}]}

        result = odetoolbox.analysis(indict, log_level="DEBUG")

        assert len(result) == 1
        assert result[0]["solver"] == "analytical"
        # the inhomogeneous update term (which depends on the timestep) is returned as a propagator, rather than inline in the update expression
        assert "__h" not in result[0]["update_expressions"]["x"]
        assert "__P__x__1" in result[0]["update_expressions"]["x"]
        assert "__h" in result[0]["propagators"]["__P__x__1"]
        assert "__h" not in result[0]["update_expressions"]["y"]
        assert "__P__y__1" not in result[0]["propagators"].keys()


class TestInhomogeneousHigherOrder:
    r"""
    Test analytic solvers for inhomogeneous ODEs of order greater than one. These are solved in the same way as first-order inhomogeneous ODEs (by the Van Loan method; see :py:meth:`SystemOfShapes._exponentiate_block`).

    Each solver is compared with a numerical reference, at generic parameter values as well as at values for which one of the singularity conditions holds (such as repeated eigenvalues, or a singular system matrix), so that the conditional solvers are tested too.
    """

    @pytest.mark.parametrize("formulation", ["second_order", "first_order_system"])
    def test_critically_damped(self, formulation: str):
        r"""Test :math:`x'' = -x / \tau^2 - 2 x' / \tau + I_e`, given either as a single second-order ODE, or as an equivalent system of two first-order ODEs."""
        if formulation == "second_order":
            dynamics = [{"expression": "x'' = -x / tau**2 - 2 * x' / tau + I_e",
                         "initial_values": {"x": "0.", "x'": "10."}}]
            derivative_name = "x__d"
        else:
            dynamics = [{"expression": "x' = y",
                         "initial_value": "0."},
                        {"expression": "y' = -x / tau**2 - 2 * y / tau + I_e",
                         "initial_value": "10."}]
            derivative_name = "y"

        solver_dict = _analyse(dynamics)
        assert set(solver_dict["state_variables"]) == {"x", derivative_name}

        for tau, I_e in [(10., -1.), (2., 3.), (2., 0.)]:
            _integrate_and_compare(solver_dict,
                                   lambda y, t: [y[1], -y[0] / tau**2 - 2 * y[1] / tau + I_e],
                                   {"x": 0., derivative_name: 10.},
                                   {"tau": tau, "I_e": I_e})

    def test_overdamped(self):
        r"""Test :math:`x'' = -ab x - (a + b) x' + I_e`, with eigenvalues :math:`-a` and :math:`-b`. This includes the cases of a repeated eigenvalue (:math:`a = b`) and of a singular system matrix (:math:`a = 0` and/or :math:`b = 0`), the latter of which yields a ramp or a parabola instead of a fixed point."""
        solver_dict = _analyse([{"expression": "x'' = -a * b * x - (a + b) * x' + I_e",
                                 "initial_values": {"x": "1.", "x'": "-1."}}])

        # only the actual singularities are found; in particular not :math:`a = -b` (from the particular solution :math:`I_e / (a + b)` of the equation for :math:`x'`), for which the system is not singular
        a_sym, b_sym = sympy.symbols("a, b", real=True)
        equalities = {relation for condition in solver_dict["conditions"].keys() if condition != "default"
                      for and_term in AnalyticIntegrator._parse_condition(condition)
                      for relation in and_term if isinstance(relation, SymmetricEq)}
        assert equalities == {SymmetricEq(a_sym, 0), SymmetricEq(b_sym, 0), SymmetricEq(a_sym, b_sym)}

        for a, b in [(.5, .2), (.5, .5), (0., .5), (.5, 0.), (0., 0.)]:
            _integrate_and_compare(solver_dict,
                                   lambda y, t: [y[1], -a * b * y[0] - (a + b) * y[1] + 2.],
                                   {"x": 1., "x__d": -1.},
                                   {"a": a, "b": b, "I_e": 2.})

    def test_constant_second_derivative(self):
        r"""Test :math:`x'' = c`, for which the system matrix is nilpotent, and check the closed-form propagators: :math:`x \leftarrow x + h x' + h^2 c / 2` and :math:`x' \leftarrow x' + h c`."""
        solver_dict = _analyse([{"expression": "x'' = c",
                                 "initial_values": {"x": "0.", "x'": "0."}}])

        h, c = sympy.symbols("__h, c", real=True)
        Qb_x = sympy.sympify(solver_dict["propagators"]["__P__x__1"], locals={"__h": h, "c": c})
        Qb_x_d = sympy.sympify(solver_dict["propagators"]["__P__x__d__1"], locals={"__h": h, "c": c})
        assert sympy.simplify(Qb_x - c * h**2 / 2) == 0
        assert sympy.simplify(Qb_x_d - c * h) == 0

        _integrate_and_compare(solver_dict,
                               lambda y, t: [y[1], 3.],
                               {"x": 0., "x__d": 0.},
                               {"c": 3.})

    def test_third_order(self):
        r"""Test the third-order ODE :math:`(D + 1/\tau)^3 x = I_e`, i.e. :math:`x''' = -x / \tau^3 - 3 x' / \tau^2 - 3 x'' / \tau + I_e`."""
        solver_dict = _analyse([{"expression": "x''' = -x / tau**3 - 3 * x' / tau**2 - 3 * x'' / tau + I_e",
                                 "initial_values": {"x": "0.", "x'": "1.", "x''": "0."}}])
        assert set(solver_dict["state_variables"]) == {"x", "x__d", "x__d__d"}

        tau = 2.
        I_e = 1.5
        _integrate_and_compare(solver_dict,
                               lambda y, t: [y[1], y[2], -y[0] / tau**3 - 3 * y[1] / tau**2 - 3 * y[2] / tau + I_e],
                               {"x": 0., "x__d": 1., "x__d__d": 0.},
                               {"tau": tau, "I_e": I_e},
                               t_max=40.)

    def test_driving_downstream_ode(self):
        r"""Test an inhomogeneous second-order ODE that drives a first-order ODE, which has no inhomogeneous term of its own: :math:`x'' = -x / \tau^2 - 2 x' / \tau + I_e`, :math:`V' = -V / \tau_m + x`. The solver is tested for :math:`\tau \neq \tau_m` as well as for :math:`\tau = \tau_m`."""
        solver_dict = _analyse([{"expression": "x'' = -x / tau**2 - 2 * x' / tau + I_e",
                                 "initial_values": {"x": "0.", "x'": "0."}},
                                {"expression": "V' = -V / tau_m + x",
                                 "initial_value": "-1."}])
        assert set(solver_dict["state_variables"]) == {"x", "x__d", "V"}

        for tau, tau_m in [(2., 10.), (2., 2.)]:
            _integrate_and_compare(solver_dict,
                                   lambda y, t: [y[1], -y[0] / tau**2 - 2 * y[1] / tau + 1., -y[2] / tau_m + y[0]],
                                   {"x": 0., "x__d": 0., "V": -1.},
                                   {"tau": tau, "tau_m": tau_m, "I_e": 1.},
                                   t_max=40.)
