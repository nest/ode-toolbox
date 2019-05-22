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
    "input_time_symbol" : "t",
    "output_timestep_symbol" : "__h"
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
    output_timestep_symbol = default_config["output_timestep_symbol"]
    if "options" in indict.keys():
        options_dict = indict["options"]
        if "output_timestep_symbol" in options_dict.keys():
            output_timestep_symbol = options_dict["output_timestep_symbol"]

    input_time_symbol = default_config["input_time_symbol"]
    if "options" in indict.keys():
        options_dict = indict["options"]
        if "input_time_symbol" in options_dict.keys():
            input_time_symbol = options_dict["input_time_symbol"]

    if "dynamics" not in indict:
        print("Warning: empty input (no dynamical equations found); returning empty output")
        solvers_json = {}
        return solvers_json

    # first run for grabbing all the variable names. Coefficients might be incorrect.
    all_variable_symbols = []
    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json, time_symbol=input_time_symbol)
        all_variable_symbols.extend(shape.get_all_variable_symbols())
    print("all_variable_symbols = " + str(all_variable_symbols))
    
    # second run provides the now-known list of variable symbols
    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json, all_variable_symbols=all_variable_symbols, time_symbol=input_time_symbol)
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
    #   generate analytical solutions (propagators) where possible
    #
    
    solvers_json = []
    analytic_solver_json = None
    syms = [ node_sym for node_sym, _node_is_lin in node_is_lin.items() if _node_is_lin ]
    if len(syms) > 0:
        print("Generating propagators for the following symbols: " + ", ".join([str(k) for k, v in node_is_lin.items() if v == True]))
        sub_sys = shape_sys.get_sub_system(syms)
        analytic_solver_json = sub_sys.generate_propagator_solver(output_timestep_symbol=output_timestep_symbol)
        analytic_solver_json["solver"] = "analytical"
        solvers_json.append(analytic_solver_json)


    #
    #   generate numerical solvers for the remainder
    #

    if len(syms) < len(shape_sys.x_):
        print("Generating numerical solver for the following symbols: " + ", ".join([str(k) for k, v in node_is_lin.items() if v == False]))
        syms = [ node_sym for node_sym, _node_is_lin in node_is_lin.items() if not _node_is_lin ]
        sub_sys = shape_sys.get_sub_system(syms)
        solver_json = sub_sys.generate_numeric_solver()
        solver_json["solver"] = "numeric"   # will be overwritten if stiffness testing is used
        if HAVE_STIFFNESS and enable_stiffness_check:
            print("Performing stiffness test...")
            kwargs = {}
            if "options" in indict.keys() and "random_seed" in indict["options"].keys():
                random_seed = int(indict["options"]["random_seed"])
                assert random_seed >= 0, "Random seed needs to be a non-negative integer"
                kwargs["random_seed"] = random_seed
            if "parameters" in indict.keys():
                kwargs["parameters"] = indict["parameters"]
            if not analytic_solver_json is None:
                kwargs["analytic_solver_dict"] = analytic_solver_json
            tester = stiffness.StiffnessTester(sub_sys, shapes, **kwargs)
            solver_type = tester.check_stiffness()
            solver_json["solver"] += "-" + solver_type
            print(solver_type + " scheme")

        #else:
            #print("(stiffness test skipped, PyGSL not available)")

        solvers_json.append(solver_json)

    #
    #   copy the initial values from the input to the output for convenience; convert to numeric values
    #

    for solver_json in solvers_json:
        solver_json["initial_values"] = {}
        for shape in shapes:
            all_shape_symbols = [ str(sympy.Symbol(str(shape.symbol) + "__d" * i)) for i in range(shape.order) ]
            for sym in all_shape_symbols:
                if sym in solver_json["state_variables"]:
                    solver_json["initial_values"][sym] = str(shape.get_initial_value(sym.replace("__d", "'")))


    #
    #   copy the parameter values from the input to the output for convenience; convert into numeric values
    #

    if "parameters" in indict.keys():
        for solver_json in solvers_json:
            solver_json["parameters"] = {}
            for param_name, param_expr in indict["parameters"].items():
                # only make parameters appear in a solver if they are actually used there
                symbol_appears_in_any_expr = False
                if "update_expressions" in solver_json.keys():
                    for sym, expr in solver_json["update_expressions"].items():
                        if param_name in [str(sym) for sym in list(expr.atoms())]:
                            symbol_appears_in_any_expr = True
                            break

                if "propagators" in solver_json.keys():
                    for sym, expr in solver_json["propagators"].items():
                        if param_name in [str(sym) for sym in list(expr.atoms())]:
                        #if len(expr.atoms(param_name)) > 0:
                            symbol_appears_in_any_expr = True
                            break
                
                if symbol_appears_in_any_expr:
                    solver_json["parameters"][param_name] = str(sympy.parsing.sympy_parser.parse_expr(param_expr, global_dict=Shape._sympy_globals).n())

    #
    #   convert expressions from sympy to string
    #
    
    for solver_json in solvers_json:
        if "update_expressions" in solver_json.keys():
            for sym, expr in solver_json["update_expressions"].items():
                solver_json["update_expressions"][sym] = str(expr)

        if "propagators" in solver_json.keys():
            for sym, expr in solver_json["propagators"].items():
                solver_json["propagators"][sym] = str(expr)

    return solvers_json

