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
import sympy
import pytest

from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.shapes import Shape
from odetoolbox.system_of_shapes import SystemOfShapes


class TestInhomogeneous:
    """Test correct propagators returned for simple inhomogeneous ODEs"""

    @pytest.mark.parametrize("ode_definition", ["(U - x) / tau",
                                                "(1 - x) / tau"])
    def test_inhomogeneous_solver(self, ode_definition):
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
        print(sys_of_shape.reconstitute_expr())
        solver_dict = sys_of_shape.generate_propagator_solver()
        solver_dict["parameters"] = parameters_dict
        print(solver_dict)

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": str(x0)})
        analytic_integrator.reset()

        dt = 1.

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

    def test_inhomogeneous_simultaneous(self):
        U = .2
        tau = 5.  # [s]

        x0 = 0.

        parameters_dict = {sympy.Symbol("U"): str(U),
                           sympy.Symbol("tau1"): str(tau),
                           sympy.Symbol("tau2"): str(tau)}

        shape_x = Shape.from_ode("x", "(U - x) / tau1", initial_values={"x": str(x0)}, parameters=parameters_dict)
        shape_y = Shape.from_ode("y", "(1 - y) / tau2", initial_values={"y": str(x0)}, parameters=parameters_dict)

        sys_of_shape = SystemOfShapes.from_shapes([shape_x, shape_y], parameters=parameters_dict)
        print(sys_of_shape.reconstitute_expr())
        solver_dict = sys_of_shape.generate_propagator_solver()
        solver_dict["parameters"] = parameters_dict
        print(solver_dict)

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"x": str(x0), "y": str(x0)})
        analytic_integrator.reset()

        dt = 1.

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

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="Only first-order inhomogeneous ODEs are supported")
    def test_inhomogeneous_solver_second_order_system(self):
        tau = 10.  # [s]
        parameters_dict = {sympy.Symbol("tau"): str(tau)}

        x0 = 0.
        x0d = 10.

        shape1 = Shape.from_ode("x", "y", initial_values={"x": str(x0)}, parameters=parameters_dict)
        shape2 = Shape.from_ode("y", "-1/tau**2 * x - 2/tau * y - 1", initial_values={"y": str(x0d)}, parameters=parameters_dict)
        sys_of_shape = SystemOfShapes.from_shapes([shape1, shape2], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="Only first-order inhomogeneous ODEs are supported")
    def test_inhomogeneous_solver_second_order(self):
        tau = 10.  # [s]
        parameters_dict = {sympy.Symbol("tau"): str(tau)}

        x0 = 0.
        x0d = 10.

        shape = Shape.from_ode("x", "-1/tau**2 * x - 2/tau * x' - 1", initial_values={"x": str(x0), "x'": str(x0d)}, parameters=parameters_dict)
        sys_of_shape = SystemOfShapes.from_shapes([shape], parameters=parameters_dict)
        solver_dict = sys_of_shape.generate_propagator_solver()
