#
# analytic.py
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

from sympy import diff, exp, Matrix, simplify, sqrt, Symbol, sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.matrices import zeros


class Propagator(object):
    """Class to store components of an exact propagation step.

    For any given linear constant coefficient ODE with an
    inhomogeneous part wich is a sum of shapes that satisfy a linear
    homogeneous ODE (i.e. instances of class `Shape`).

    The basic idea is to reformulate the ODE or systems of ODEs as the
    ODE y' = Ay and to calculate A and then exp(Ah) to give an
    evolution of the system for a given timestep `h`.

    Example:
    ========
    shape_alpha = ShapeFunction("shape_alpha", "e / tau * t * exp(-t / tau)")
    shape_exp = ShapeFunction("shape_exp", "exp(-t/tau)")
    shape_sin = ShapeFunction("shape_sinb", "sin(t)")
    shapes = [shape_alpha, shape_exp, shape_sin]

    ode_symbol = "V_m"
    ode_definition = "-1/Tau * V_m-1/C * (shape_alpha + shape_exp + shape_sin + currents + I_E)"
    prop_matrices, const_input, step_const = otpm.ode_to_prop_matrices(ode_symbol, ode_definition, shapes)

    """

    def __init__(self, ode_symbol, ode_definition, shapes, timestep_symbol_name="__h"):

        self.ode_symbol = parse_expr(ode_symbol)
        self.ode_definition = parse_expr(ode_definition)
        self.propagator_matrices = []
        self.step_constant = Symbol("")
        self.propagator = None
        self.ode_updates = None
        self._timestep_symbol = Symbol(timestep_symbol_name)
        constant_input, step_constant = self._compute_propagator_matrices(shapes)
        self._compute_propagation_step(shapes, constant_input, step_constant)

    def _compute_propagator_matrices(self, shapes):
        """Compute the propagator matrices using the given shapes.

        For a differential equation
        
            V'= 1/Tau * V + 1/C * shape
        
        the factor of the ode symbol (`ode_symbol_factor`) would be
        `1/Tau`, the factor of the shape (`shape_factor`) `1/C`. As we
        process a list of shape, we get a list of `shape_factor`s.

        The shape factors in the ODE and the derivative factors of the
        shape are used to set up the propagator matrices.

        For shapes that satisfy a homogeneous linear ODE of order 1 or
        2, we create an lower triangular matrices as this allows a
        more efficient specification of the update step in
        compute_update_step().

        For shapes that satisfy a homogeneous linear ODE of an order
        larger than 2, we calculate A by choosing the state variables
        canonically as y_0 = shape^(n), ..., y_{n-1} = shape, y_n = V
        """

        # Initialize the factor in front of the ode symbol, an empty
        # list to hold the factors for the shapes and a symbol to
        # represent the time step.
        ode_symbol_factor = diff(self.ode_definition, self.ode_symbol)
        shape_factors = []
        h = self._timestep_symbol

        if not shapes:
            print("ERROR: no shapes given, unable to calculate the propagator matrix.")
            import sys
            sys.exit(1)

        for shape in shapes:

            shape_factor = diff(self.ode_definition, shape.symbol)

            if shape.order == 1:
                A = Matrix([[shape.derivative_factors[0], 0],
                            [shape_factor, ode_symbol_factor]])
            elif shape.order == 2:
                solutionpq = -shape.derivative_factors[1] / 2 + \
                             sqrt(shape.derivative_factors[1]**2 / 4 + \
                                  shape.derivative_factors[0])
                A = Matrix([[shape.derivative_factors[1]+solutionpq, 0, 0 ],
                            [1, -solutionpq, 0 ],
                            [0, shape_factor, ode_symbol_factor]])
            else:
                A = zeros(shape.order + 1)
                A[shape.order, shape.order] = ode_symbol_factor
                A[shape.order, shape.order - 1] = shape_factor
                for j in range(0, shape.order):
                    A[0, j] = shape.derivative_factors[shape.order - j - 1]
                for i in range(1, shape.order):
                    A[i, i - 1] = 1
    
            shape_factors.append(shape_factor)

            self.propagator_matrices.append(simplify(exp(A * h)))
    
        step_constant = -1/ode_symbol_factor * (1 - exp(h * ode_symbol_factor))
        
        constant_input = self.ode_definition - ode_symbol_factor * self.ode_symbol
        for shape_factor, shape in zip(shape_factors, shapes):
            constant_input -= shape_factor * shape.symbol

        return simplify(constant_input), simplify(step_constant)

    def _compute_propagation_step(self, shapes, constant_input, step_constant):
        """Compute a calculation specification for the update step.

        """

        ode_symbol_factor_h = self.propagator_matrices[0][shapes[0].order, shapes[0].order]
        constant_term = "(" + str(ode_symbol_factor_h) + ") * " + str(self.ode_symbol) + \
                        "+ (" + str(constant_input) + ") * (" + str(step_constant) + ")"

        self.propagator = {}
        self.ode_updates = [str(self.ode_symbol) + " = " + constant_term]
        for p, shape in zip(self.propagator_matrices, shapes):
            P = zeros(shape.order + 1, shape.order + 1)
            for i in range(shape.order + 1):
                for j in range(shape.order + 1):
                    if simplify(p[i, j]) != sympify(0):
                        symbol_p_i_j = "__P_{}__{}_{}".format(shape.symbol, i, j)
                        P[i, j] = parse_expr(symbol_p_i_j)
                        self.propagator[symbol_p_i_j] = str(p[i, j])
    
            y = zeros(shape.order + 1, 1)
            for i in range(shape.order):
                y[i] = shape.state_variables[i]
            y[shape.order] = self.ode_symbol
    
            P[shape.order, shape.order] = 0
            z = P * y
    
            self.ode_updates.append(str(self.ode_symbol) + " += " + str(z[shape.order]))
    
            shape.state_updates = P[:shape.order, :shape.order] * y[:shape.order, 0]


def compute_analytical_solution(ode_symbol, ode_definition, shapes, timestep_symbol_name="__h"):

    propagator = Propagator(ode_symbol, ode_definition, shapes, timestep_symbol_name=timestep_symbol_name)

    data = {
        "solver": "analytical",
        "ode_updates": propagator.ode_updates,
        "propagator": propagator.propagator,
        "shape_initial_values": [],
        "shape_state_updates": [],
        "shape_state_variables": [],
    }

    for shape in shapes:
        data["shape_initial_values"].append([str(x) for x in shape.initial_values])
        data["shape_state_updates"].append([str(x) for x in shape.state_updates])
        data["shape_state_variables"].append([str(x) for x in shape.state_variables])

    return data
