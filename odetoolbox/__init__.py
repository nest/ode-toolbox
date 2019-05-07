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

from .shapes import Shape
from .analytic import Propagator

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

    print("Performing dependency analysis...")
    from graphviz import Digraph
    E = []
    for shape1 in shapes:
        for shape2 in shapes:

            # check if symb1 occurs in the expression for symb2
            shape2_depends_on_shape1 = shape2.diff_rhs_derivatives.has(shape1.symbol)

            if not shape2_depends_on_shape1:
                for derivative_factor in shape2.derivative_factors:
                    if derivative_factor.has(shape1.symbol):
                        # shape 2 depends on shape 1
                        shape2_depends_on_shape1 = True
                        break

            if shape2_depends_on_shape1:
                E.append((str(shape2.symbol), str(shape1.symbol)))

    dot = Digraph(comment="Dependency graph", engine="fdp")#, format="pdf")
    dot.attr(compound="true")
    nodes = []
    for shape in shapes:
        if shape.is_lin_const_coeff():
            style = "filled"
            colour = "chartreuse"
        else:
            style = "rounded"
            colour = "black"
        if shape.order > 1:
            with dot.subgraph(name="cluster_" + str(shape.symbol)) as sg:
                nodes.append("cluster_" + str(shape.symbol))
                sg.attr(label=str(shape.symbol))
                for i in range(shape.order):
                    sg.node(str(shape.symbol) + i * "'", style=style, color=colour)#, str(shape.symbol) + str(i))
                    print("Creating sg node for " + str(shape.symbol) + i * "'" + ", colour = " + str(colour))
        else:
            dot.node(str(shape.symbol), style=style, color=colour)
            nodes.append(str(shape.symbol))
            print("Creating order 1 node for " + str(shape.symbol) + ", colour = " + str(colour))

    for e in E:
        prefer_connections_to_clusters = False
        if prefer_connections_to_clusters:
            e = list(e)
            for i in range(2):
                if "cluster_" +e[i] in nodes:
                    e[i] = "cluster_" + e[i]

        print("Edge from " + str(e[0]) + " to " + str(e[1]))
        dot.edge(str(e[0]), str(e[1]))
    #dot.view()
    dot.render("/tmp/remotefs/ode_dependency_graph.dot")


    print("Generating solvers...")
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
