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

import numpy as np
import pytest
import sympy

import odetoolbox

from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.shapes import Shape
from odetoolbox.system_of_shapes import PropagatorGenerationException, SystemOfShapes


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
            state_ = analytic_integrator.get_value(t)["x"]
            actual.append(state_)

            cur_x = x0 + 42 * t
            correct.append(cur_x)

        np.testing.assert_allclose(correct, actual)

    @pytest.mark.parametrize("dt", [.1, 1.])
    @pytest.mark.parametrize("ode_definition", ["(U - x) / tau",
                                                "(1 - x) / tau"])
    def test_inhomogeneous_solver(self, dt, ode_definition):
        U = .2
        if ode_definition == "(1 - x) / tau":
            U = 1.
        tau = 5.  # [s]

        x0 = 0.

        parameters_dict = {sympy.Symbol("U"): str(U),
                           sympy.Symbol("tau"): str(tau)}

        shape = Shape.from_ode("x", ode_definition, initial_values={"x": str(x0)}, parameters=parameters_dict)

        assert not shape.is_homogeneous()
        assert shape.is_lin_const_coeff()

        sys_of_shape = SystemOfShapes.from_shapes([shape], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()
        solver_dict["parameters"] = parameters_dict

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": str(x0)})
        analytic_integrator.reset()

        actual = []
        correct = []
        cur_x = x0
        timevec = np.arange(0., 100., dt)
        kernel = np.exp(-dt / tau)
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)["x"]
            actual.append(state_)
            correct.append(cur_x)
            cur_x = U + kernel * (cur_x - U)

        np.testing.assert_allclose(correct, actual)

    @pytest.mark.parametrize("dt", [.1, 1.])
    def test_inhomogeneous_simultaneous(self, dt: float):
        U = .2
        tau = 5.  # [s]

        x0 = 0.

        parameters_dict = {sympy.Symbol("U"): str(U),
                           sympy.Symbol("tau1"): str(tau),
                           sympy.Symbol("tau2"): str(tau)}

        shape_x = Shape.from_ode("x", "(U - x) / tau1", initial_values={"x": str(x0)}, parameters=parameters_dict)
        shape_y = Shape.from_ode("y", "(1 - y) / tau2", initial_values={"y": str(x0)}, parameters=parameters_dict)

        sys_of_shape = SystemOfShapes.from_shapes([shape_x, shape_y], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()
        solver_dict["parameters"] = parameters_dict

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
            state_x = analytic_integrator.get_value(t)["x"]
            state_y = analytic_integrator.get_value(t)["y"]
            actual_x.append(state_x)
            actual_y.append(state_y)
            correct_x.append(cur_x)
            correct_y.append(cur_y)
            cur_x = U + kernel * (cur_x - U)
            cur_y = 1 + kernel * (cur_y - 1)

        np.testing.assert_allclose(correct_x, actual_x)
        np.testing.assert_allclose(correct_y, actual_y)

    @pytest.mark.xfail(strict=True, raises=PropagatorGenerationException, reason="Only first-order inhomogeneous ODEs are supported")
    def test_inhomogeneous_solver_second_order(self):
        r"""test failure to generate propagators for inhomogeneous 2nd order ODE"""
        tau = 10.  # [s]
        parameters_dict = {sympy.Symbol("tau"): str(tau)}

        x0 = 0.
        x0d = 10.

        shape = Shape.from_ode("x", "-1/tau**2 * x - 2/tau * x' - 1", initial_values={"x": str(x0), "x'": str(x0d)}, parameters=parameters_dict)
        sys_of_shape = SystemOfShapes.from_shapes([shape], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()

    @pytest.mark.xfail(strict=True, raises=PropagatorGenerationException, reason="Only first-order inhomogeneous ODEs are supported")
    def test_inhomogeneous_solver_second_order_system(self):
        r"""test failure to generate propagators for inhomogeneous 2nd order ODE"""
        tau = 10.  # [s]
        parameters_dict = {sympy.Symbol("tau"): str(tau)}

        x0 = 0.
        x0d = 10.

        shape1 = Shape.from_ode("x", "y", initial_values={"x": str(x0)}, parameters=parameters_dict)
        shape2 = Shape.from_ode("y", "-1/tau**2 * x - 2/tau * y - 1", initial_values={"y": str(x0d)}, parameters=parameters_dict)
        sys_of_shape = SystemOfShapes.from_shapes([shape1, shape2], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()

    def test_inhomogeneous_solver_second_order_system_api(self):
        r"""test failure to generate propagators when called via analysis()"""
        indict = {"dynamics": [{"expression": "x' = y",
                                "initial_value": "0."},
                               {"expression": "y' = -1/tau**2 * x - 2/tau * y - 1",
                                "initial_value": "0."}],
                  "parameters": {"tau": "10."}}

        result = odetoolbox.analysis(indict, disable_stiffness_check=True)
        assert len(result) == 1 \
               and result[0]["solver"].startswith("numeric")

    def test_inhomogeneous_solver_second_order_combined_system(self):
        r"""test propagators generation for combined homogeneous/inhomogeneous ODEs"""
        tau = 10.  # [s]
        E_L = -70.  # [mV]
        parameters_dict = {sympy.Symbol("tau"): str(tau),
                           sympy.Symbol("E_L"): str(E_L)}

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
        assert "__h" in result[0]["update_expressions"]["x"]
        assert "__h" not in result[0]["update_expressions"]["y"]
