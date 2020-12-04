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

from .sympy_printer import SympyPrinter
from .system_of_shapes import SystemOfShapes
from .shapes import Shape

import copy
import json
import logging

import sympy
sympy.Basic.__str__ = lambda self: SympyPrinter().doprint(self)

try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError as ie:
    print("Warning: PyGSL is not available. The stiffness test will be skipped.")
    print("Warning: " + str(ie), end="\n\n\n")
    PYGSL_AVAILABLE = False

if PYGSL_AVAILABLE:
    from .stiffness import StiffnessTester

try:
    import graphviz
    PLOT_DEPENDENCY_GRAPH = True
except ImportError:
    PLOT_DEPENDENCY_GRAPH = False

if PLOT_DEPENDENCY_GRAPH:
    from .dependency_graph_plotter import DependencyGraphPlotter


default_config = {
    "input_time_symbol": "t",
    "output_timestep_symbol": "__h",
    "differential_order_symbol": "__d",
    "sim_time": 100E-3,
    "max_step_size": 999.,
    "integration_accuracy_abs": 1E-6,
    "integration_accuracy_rel": 1E-6
}


def _dependency_analysis(shape_sys, shapes, differential_order_symbol):
    r"""
    Perform dependency analysis and plot dependency graph.
    """
    logging.info("Dependency analysis...")
    dependency_edges = shape_sys.get_dependency_edges()
    node_is_lin = shape_sys.get_lin_cc_symbols(dependency_edges, differential_order_symbol=differential_order_symbol)
    if PLOT_DEPENDENCY_GRAPH:
        DependencyGraphPlotter.plot_graph(shapes, dependency_edges, node_is_lin, fn="/tmp/ode_dependency_graph_lin_cc.dot")
    node_is_lin = shape_sys.propagate_lin_cc_judgements(node_is_lin, dependency_edges)
    if PLOT_DEPENDENCY_GRAPH:
        DependencyGraphPlotter.plot_graph(shapes, dependency_edges, node_is_lin, fn="/tmp/ode_dependency_graph_analytically_solvable.dot")
    return dependency_edges, node_is_lin


def _read_global_config(indict, default_config):
    r"""
    Process global configuration options.
    """
    logging.info("Processing global options...")
    options_dict = copy.deepcopy(default_config)
    if "options" in indict.keys():
        for key, value in indict["options"].items():
            assert key in default_config.keys(), "Unknown key specified in global options dictionary: \"" + str(key) + "\""
            options_dict[key] = value

    return options_dict


def _from_json_to_shapes(indict, options_dict):
    r"""
    Process the input, construct Shape instances.

    :param indict: ODE-toolbox input dictionary.
    :param options_dict: ODE-toolbox global configuration dictionary.
    """

    logging.info("Processing input shapes...")
    shapes = []
    # first run for grabbing all the variable names. Coefficients might be incorrect.
    all_variable_symbols = []
    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json, time_symbol=options_dict["input_time_symbol"], differential_order_symbol=options_dict["differential_order_symbol"])
        all_variable_symbols.extend(shape.get_state_variables())
    logging.debug("From first run: all_variable_symbols = " + str(all_variable_symbols))

    # second run with the now-known list of variable symbols
    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json, all_variable_symbols=all_variable_symbols, time_symbol=options_dict["input_time_symbol"], _debug=True)
        shapes.append(shape)

    return shapes


def _analysis(indict, disable_stiffness_check=False, disable_analytic_solver=False, debug=False):
    """
    Like analysis(), but additionally returns ``shape_sys`` and ``shapes``.

    For internal use only!
    """

    # import sys;sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

    global default_config

    _init_logging(debug)

    logging.info("ode-toolbox: analysing input:")
    logging.info(json.dumps(indict, indent=4, sort_keys=True))

    if "dynamics" not in indict:
        logging.info("Warning: empty input (no dynamical equations found); returning empty output")
        solvers_json = {}
        return solvers_json

    options_dict = _read_global_config(indict, default_config)
    shapes = _from_json_to_shapes(indict, options_dict)
    shape_sys = SystemOfShapes.from_shapes(shapes)
    dependency_edges, node_is_lin = _dependency_analysis(shape_sys, shapes, differential_order_symbol=options_dict["differential_order_symbol"])


    #
    #   generate analytical solutions (propagators) where possible
    #

    solvers_json = []
    analytic_solver_json = None
    if disable_analytic_solver:
        analytic_syms = []
    else:
        analytic_syms = [node_sym for node_sym, _node_is_lin in node_is_lin.items() if _node_is_lin]

    if analytic_syms:
        logging.info("Generating propagators for the following symbols: " + ", ".join([str(k) for k in analytic_syms]))
        sub_sys = shape_sys.get_sub_system(analytic_syms)
        analytic_solver_json = sub_sys.generate_propagator_solver(output_timestep_symbol=options_dict["output_timestep_symbol"])
        analytic_solver_json["solver"] = "analytical"
        solvers_json.append(analytic_solver_json)


    #
    #   generate numerical solvers for the remainder
    #

    if len(analytic_syms) < len(shape_sys.x_):
        numeric_syms = list(set(shape_sys.x_) - set(analytic_syms))
        logging.info("Generating numerical solver for the following symbols: " + ", ".join([str(sym) for sym in numeric_syms]))
        sub_sys = shape_sys.get_sub_system(numeric_syms)
        solver_json = sub_sys.generate_numeric_solver()
        solver_json["solver"] = "numeric"   # will be appended to if stiffness testing is used
        if not disable_stiffness_check:
            if not PYGSL_AVAILABLE:
                raise Exception("Stiffness test requested, but PyGSL not available")

            logging.info("Performing stiffness test...")
            kwargs = {}
            if "options" in indict.keys() and "random_seed" in indict["options"].keys():
                random_seed = int(indict["options"]["random_seed"])
                assert random_seed >= 0, "Random seed needs to be a non-negative integer"
                kwargs["random_seed"] = random_seed
            if "parameters" in indict.keys():
                kwargs["parameters"] = indict["parameters"]
            if "stimuli" in indict.keys():
                kwargs["stimuli"] = indict["stimuli"]
            for key in ["sim_time", "max_step_size", "integration_accuracy_abs", "integration_accuracy_rel"]:
                if "options" in indict.keys() and key in options_dict.keys():
                    kwargs[key] = float(options_dict[key])
            if not analytic_solver_json is None:
                kwargs["analytic_solver_dict"] = analytic_solver_json
            tester = StiffnessTester(sub_sys, shapes, **kwargs)
            solver_type = tester.check_stiffness()
            if not solver_type is None:
                solver_json["solver"] += "-" + solver_type
                logging.info(solver_type + " scheme")

        solvers_json.append(solver_json)


    #
    #   copy the initial values from the input to the output for convenience; convert to numeric values
    #

    for solver_json in solvers_json:
        solver_json["initial_values"] = {}
        for shape in shapes:
            all_shape_symbols = [str(sympy.Symbol(str(shape.symbol) + options_dict["differential_order_symbol"] * i)) for i in range(shape.order)]
            for sym in all_shape_symbols:
                if sym in solver_json["state_variables"]:
                    solver_json["initial_values"][sym] = str(shape.get_initial_value(sym.replace(options_dict["differential_order_symbol"], "'")))


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

    logging.info("In ode-toolbox: returning outdict = ")
    logging.info(json.dumps(solvers_json, indent=4, sort_keys=True))

    return solvers_json, shape_sys, shapes


def _init_logging(debug: bool):
    """
    Initialise message logging.

    :param debug: Set to :python:`True` to increase the verbosity.
    """
    fmt = '%(levelname)s:%(message)s'
    if debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)


def analysis(indict, disable_stiffness_check: bool=False, disable_analytic_solver: bool=False, debug: bool=False):
    r"""
    The main entry point of the ODE-toolbox API.

    :param indict: Input dictionary for the analysis. For details, see https://ode-toolbox.readthedocs.io/en/latest/index.html#input
    :param disable_stiffness_check: Whether to perform stiffness checking.
    :param disable_analytic_solver: Set to True to return numerical solver recommendations, and no propagators, even for ODEs that are analytically tractable.

    :return: The result of the analysis. For details, see https://ode-toolbox.readthedocs.io/en/latest/index.html#output
    """
    d, _, _ = _analysis(indict, disable_stiffness_check=disable_stiffness_check, disable_analytic_solver=disable_analytic_solver, debug=debug)
    return d
