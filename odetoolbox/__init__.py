#
# __init__.py
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

from __future__ import print_function

from sympy import diff, simplify
from sympy.parsing.sympy_parser import parse_expr
from .system_of_shapes import SystemOfShapes

from .shapes import Shape
from .analytic import Propagator
from .dependency_graph_plotter import DependencyGraphPlotter

try:
    from . import stiffness
    HAVE_STIFFNESS = True
except:
    HAVE_STIFFNESS = False


class MalformedInput(Exception): pass

class ShapeNotLinHom(Exception): pass


default_config = {
    "input_timestep_symbol_name" : "t",
    "output_timestep_symbol_name" : "__h"
}

def analysis(indict, enable_stiffness_check=True):
    """The main entry point of the analysis.

    This function expects a single dictionary with the keys `odes`,
    `parameters` and `shapes` that describe the input to the analysis.

    The exact format of the input entries is described in the file
    `README.md`.

    :return: The result of the analysis, again as a dictionary.
    """

    #from odetoolbox.analytic import compute_analytical_solution
    #from odetoolbox.numeric import compute_numeric_solution
    from odetoolbox.shapes import Shape

    shapes = []

    print("Processing input shapes...")

    if "dynamics" not in indict:
        print("Warning: empty input (no dynamical equations found); returning empty output")
        outdict = {}
        return outdict

    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json)
        shapes.append(shape)

    print("Constructing system matrix...")
    shape_sys = SystemOfShapes(shapes)

    print("Plotting dependency graph...")
    DependencyGraphPlotter.plot_graph(shapes, shape_sys, fn="/tmp/remotefs/ode_dependency_graph.dot")
    
    
    
    output_timestep_symbol_name = default_config["output_timestep_symbol_name"]
    if "options" in indict.keys():
        options_dict = indict["options"]
        if "output_timestep_symbol_name" in options_dict.keys():
            output_timestep_symbol_name = options_dict["output_timestep_symbol_name"]
    prop = Propagator.from_shapes(shapes, output_timestep_symbol_name=output_timestep_symbol_name)
    if False:
        if False:
            print(": numerical ", end="")
            output = compute_numeric_solution(shapes)
            if HAVE_STIFFNESS and enable_stiffness_check:

                # TODO: check what happens/has to happen for shapes
                # that already have a definition of either type
                # `function` or `ode`.

                indict["shapes"] = []
                for shape in shapes:
                    ode_shape = {"type": "ode",
                                 "symbol": str(shape.symbol),
                                 "initial_values": shape.initial_values,
                                 "definition": str(shape.ode_definition)}
                    indict["shapes"].append(ode_shape)

                tester = stiffness.StiffnessTester(indict)

                # TODO: Check whether we need to run this with
                # arguments different from the defaults.
                solver_type = tester.check_stiffness()

                output["solver"] += "-" + solver_type
                print(solver_type + " scheme")

            else:
                print("(stiffness test skipped, PyGSL not available)")

    return output
