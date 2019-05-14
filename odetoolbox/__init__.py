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

import sympy

from .system_of_shapes import SystemOfShapes

from .shapes import Shape
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

    #
    #   process the input, construct Shape instances
    #

    print("Processing input shapes...")
    output_timestep_symbol_name = default_config["output_timestep_symbol_name"]
    if "options" in indict.keys():
        options_dict = indict["options"]
        if "output_timestep_symbol_name" in options_dict.keys():
            output_timestep_symbol_name = options_dict["output_timestep_symbol_name"]

    if "dynamics" not in indict:
        print("Warning: empty input (no dynamical equations found); returning empty output")
        solvers_json = {}
        return solvers_json

    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json)
        shapes.append(shape)


    #
    #   construct global system matrix
    #

    print("Constructing system matrix...")
    shape_sys = SystemOfShapes.from_shapes(shapes)


    #
    #   perform dependency analysis, plot dependency graph
    #

    print("Dependency analysis...")
    dependency_edges = shape_sys.get_dependency_edges()
    node_is_lin = shape_sys.get_lin_cc_symbols(dependency_edges)
    DependencyGraphPlotter.plot_graph(shapes, dependency_edges, node_is_lin, fn="/tmp/remotefs/ode_dependency_graph_lin_cc.dot")
    node_is_lin = shape_sys.propagate_lin_cc_judgements(node_is_lin, dependency_edges)
    DependencyGraphPlotter.plot_graph(shapes, dependency_edges, node_is_lin, fn="/tmp/remotefs/ode_dependency_graph_analytically_solvable.dot")
    
    
    #
    #   compute analytical solutions (propagators) where possible
    #
    
    solvers_json = []

    syms = [ node_sym for node_sym, _node_is_lin in node_is_lin.items() if _node_is_lin ]
    if len(syms) > 0:
        print("Generating propagators for the following symbols: " + ", ".join([str(k) for k, v in node_is_lin.items() if v == True]))

        sub_sys = shape_sys.get_sub_system(syms)
        solver_json = sub_sys.compute_propagator(output_timestep_symbol_name=output_timestep_symbol_name)
        solver_json["solver"] = "analytical"
        #solver_json = {"solver": "analytical",
                     #"propagator": prop }
        solvers_json.append(solver_json)

    
    #
    #   compute numerical solvers for the remainder
    #

    #if len(syms) < len(shape_sys.x_):
        #print("Picking numerical solver for the following symbols: " + ", ".join([str(k) for k, v in node_is_lin.items() if v == False]))
        #solver_json = sub_sys.compute_numeric_solution(shapes)
        #if HAVE_STIFFNESS and enable_stiffness_check:
            #tester = stiffness.StiffnessTester(shapes)
            #solver_type = tester.check_stiffness()

            #output["solver"] += "-" + solver_type
            #print(solver_type + " scheme")

        #else:
            #print("(stiffness test skipped, PyGSL not available)")

        #solvers_json.append(solver_json)

    #
    #   copy the initial values from the input to the output for convenience
    #

    for solver_json in solvers_json:
        solver_json["initial_values"] = {}
        for shape in shapes:
            all_shape_symbols = [ str(sympy.Symbol(str(shape.symbol) + "__d" * i)) for i in range(shape.order) ]
            for sym in all_shape_symbols:
                if sym in solver_json["state_variables"]:
                    solver_json["initial_values"][sym] = str(shape.get_initial_value(sym.replace("__d", "'")))
    import pdb;pdb.set_trace()

    return solvers_json

