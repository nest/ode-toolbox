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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import json
import logging
import sys
import sympy
from sympy.core.expr import Expr as SympyExpr

from .config import Config
from .sympy_helpers import _check_numerical_issue, _check_forbidden_name, _find_in_matrix, _is_zero, _is_sympy_type, SympyPrinter
from .system_of_shapes import SystemOfShapes
from .shapes import MalformedInputException, Shape


try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError as ie:
    logging.warning("PyGSL is not available. The stiffness test will be skipped.")
    logging.warning("Error when importing: " + str(ie))
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

sympy.Basic.__str__ = lambda self: SympyPrinter().doprint(self)


def _find_analytically_solvable_equations(shape_sys, shapes, parameters=None):
    r"""
    Find which equations can be solved analytically (and, conversely, which cannot).

    Perform dependency analysis and plot dependency graph.
    """
    logging.info("Finding analytically solvable equations...")
    dependency_edges = shape_sys.get_dependency_edges()

    if PLOT_DEPENDENCY_GRAPH:
        node_is_analytically_solvable = {sym: False for sym in list(shape_sys.x_)}
        DependencyGraphPlotter.plot_graph(shapes, dependency_edges, node_is_analytically_solvable, fn="/tmp/ode_dependency_graph.dot")

    node_is_analytically_solvable = shape_sys.get_lin_cc_symbols(dependency_edges, parameters=parameters)

    if PLOT_DEPENDENCY_GRAPH:
        DependencyGraphPlotter.plot_graph(shapes, dependency_edges, node_is_analytically_solvable, fn="/tmp/ode_dependency_graph_analytically_solvable_before_propagated.dot")

    # cannot analytically solve inhomogeneous, order > 1 shapes
    for i in range(len(shape_sys.x_)):
        if not _is_zero(shape_sys.b_[i]) and shape_sys.shape_order_from_system_matrix(i) > 1 and shape_sys.x_[i] in shape_sys.get_connected_symbols(i):
            node_is_analytically_solvable[shape_sys.x_[i]] = False

        for j in range(len(shape_sys.x_)):
            if not i == j and not _is_zero(shape_sys.A_[i, j]) and not _is_zero(shape_sys.b_[_find_in_matrix(shape_sys.x_, shape_sys.x_[j])]):
                # this shape depends on another ODE that is inhomogeneous -- can't be solved analytically by this version of ODE-toolbox
                node_is_analytically_solvable[shape_sys.x_[i]] = False

    node_is_analytically_solvable = shape_sys.propagate_lin_cc_judgements(node_is_analytically_solvable, dependency_edges)
    if PLOT_DEPENDENCY_GRAPH:
        DependencyGraphPlotter.plot_graph(shapes, dependency_edges, node_is_analytically_solvable, fn="/tmp/ode_dependency_graph_analytically_solvable.dot")

    return dependency_edges, node_is_analytically_solvable


def _read_global_config(indict):
    r"""
    Process global configuration options.
    """
    logging.info("Processing global options...")
    if "options" in indict.keys():
        for key, value in indict["options"].items():
            assert key in Config.config.keys(), "Unknown key specified in global options dictionary: \"" + str(key) + "\""
            Config.config[key] = value


def _from_json_to_shapes(indict, parameters=None) -> Tuple[List[Shape], Dict[sympy.Symbol, str]]:
    r"""
    Process the input, construct Shape instances.

    :param indict: ODE-toolbox input dictionary.
    """

    logging.info("Processing input shapes...")

    # first run for grabbing all the variable names. Coefficients might be incorrect.
    all_variable_symbols = []
    all_parameter_symbols = set()
    all_variable_symbols_ = set()
    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json, parameters=parameters)
        all_variable_symbols.extend(shape.get_state_variables())
        all_variable_symbols_.update(shape.get_state_variables(derivative_symbol=Config().differential_order_symbol))
        all_parameter_symbols.update(set(shape.reconstitute_expr().free_symbols))
    all_parameter_symbols -= all_variable_symbols_
    del all_variable_symbols_
    assert all([_is_sympy_type(sym) for sym in all_variable_symbols])
    logging.info("All known variables: " + str(all_variable_symbols) + ", all parameters used in ODEs: " + str(all_parameter_symbols))

    # validate input for forbidden names
    for var in set(all_variable_symbols) | all_parameter_symbols:
        _check_forbidden_name(var)

    # validate parameters
    for param in all_parameter_symbols:
        if parameters is None:
            parameters = dict()

        assert isinstance(param, SympyExpr)
        if not param in parameters.keys():
            # this parameter was used in an ODE, but not explicitly numerically specified
            logging.info("No numerical value specified for parameter \"" + str(param) + "\"")    # INFO level because this is OK!
            parameters[param] = None

    # second run with the now-known list of variable symbols
    shapes = []
    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json, all_variable_symbols=all_variable_symbols, parameters=parameters, _debug=True)
        shapes.append(shape)

    return shapes, parameters


def _find_variable_definition(indict, name: str, order: int) -> Optional[str]:
    r"""Find the definition (as a string in the input dictionary) of variable named ``name`` with order ``order``, and return it as a string. Return None if a definition by that name and order could not be found."""
    for dyn in indict["dynamics"]:
        if "expression" in dyn.keys():
            exprs = [dyn["expression"]]
        elif "expressions" in dyn.keys():
            exprs = dyn["expressions"]

        for expr in exprs:
            name_, order_, rhs = Shape._parse_defining_expression(expr)
            if name_ == name and order_ == order:
                return rhs

    return None


def _get_all_first_order_variables(indict) -> Iterable[str]:
    r"""Return a list of variable names, containing those variables that were defined as a first-order ordinary differential equation in the input."""
    variable_names = []

    for dyn in indict["dynamics"]:
        if "expression" in dyn.keys():
            exprs = [dyn["expression"]]
        elif "expressions" in dyn.keys():
            exprs = dyn["expressions"]

        for expr in exprs:
            name, order, rhs = Shape._parse_defining_expression(expr)
            if order == 1:
                variable_names.append(name)

    return variable_names


def _analysis(indict, disable_stiffness_check: bool = False, disable_analytic_solver: bool = False, preserve_expressions: Union[bool, Iterable[str]] = False, simplify_expression: Optional[str] = None, log_level: Union[str, int] = logging.WARNING) -> Tuple[List[Dict], SystemOfShapes, List[Shape]]:
    r"""
    Like analysis(), but additionally returns ``shape_sys`` and ``shapes``.

    For internal use only!
    """

    # import sys;sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

    _init_logging(log_level)

    logging.info("Analysing input:")
    logging.info(json.dumps(indict, indent=4, sort_keys=True))

    if "dynamics" not in indict:
        logging.info("Warning: empty input (no dynamical equations found); returning empty output")
        return [], SystemOfShapes.from_shapes([]), []

    _read_global_config(indict)

    if simplify_expression:
        Config.config["simplify_expression"] = simplify_expression

    # copy parameters from the input and make sure keys are of type sympy.Symbol
    parameters = None
    if "parameters" in indict.keys():
        parameters = {}
        for k, v in indict["parameters"].items():
            if type(k) is str:
                parameters[sympy.Symbol(k)] = v
            else:
                assert type(k) is sympy.Symbol
                parameters[k] = v

            _check_forbidden_name(k)


    #
    #   create Shapes and SystemOfShapes
    #

    shapes, parameters = _from_json_to_shapes(indict, parameters=parameters)

    for shape in shapes:
        if not shape.is_homogeneous() and shape.order > 1:
            logging.error("For symbol " + str(shape.symbol) + ": higher-order inhomogeneous ODEs are not supported")
            sys.exit(1)

    shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
    _, node_is_analytically_solvable = _find_analytically_solvable_equations(shape_sys, shapes, parameters=parameters)


    #
    #   generate analytical solutions (propagators) where possible
    #

    solvers_json = []
    if disable_analytic_solver:
        analytic_syms = []
    else:
        analytic_syms = [node_sym for node_sym, _node_is_analytically_solvable in node_is_analytically_solvable.items() if _node_is_analytically_solvable]

    analytic_solver_json = None
    if analytic_syms:
        logging.info("Generating propagators for the following symbols: " + ", ".join([str(k) for k in analytic_syms]))
        sub_sys = shape_sys.get_sub_system(analytic_syms)
        analytic_solver_json = sub_sys.generate_propagator_solver()
        analytic_solver_json["solver"] = "analytical"
        solvers_json.append(analytic_solver_json)


    #
    #   generate numerical solvers for the remainder
    #

    if len(analytic_syms) < len(shape_sys.x_):
        numeric_syms = list(set(shape_sys.x_) - set(analytic_syms))
        logging.info("Generating numerical solver for the following symbols: " + ", ".join([str(sym) for sym in numeric_syms]))
        sub_sys = shape_sys.get_sub_system(numeric_syms)
        solver_json = sub_sys.generate_numeric_solver(state_variables=shape_sys.x_)
        solver_json["solver"] = "numeric"   # will be appended to if stiffness testing is used
        if not disable_stiffness_check:
            if not PYGSL_AVAILABLE:
                raise Exception("Stiffness test requested, but PyGSL not available")

            logging.info("Performing stiffness test...")
            kwargs = {}   # type: Dict[str, Any]
            if "options" in indict.keys() and "random_seed" in indict["options"].keys():
                random_seed = int(indict["options"]["random_seed"])
                assert random_seed >= 0, "Random seed needs to be a non-negative integer"
                kwargs["random_seed"] = random_seed
            if "parameters" in indict.keys():
                kwargs["parameters"] = indict["parameters"]
            if "stimuli" in indict.keys():
                kwargs["stimuli"] = indict["stimuli"]
            for key in ["sim_time", "max_step_size", "integration_accuracy_abs", "integration_accuracy_rel"]:
                if "options" in indict.keys() and key in Config().keys():
                    kwargs[key] = float(Config()[key])
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
            all_shape_symbols = [str(sympy.Symbol(str(shape.symbol) + Config().differential_order_symbol * i)) for i in range(shape.order)]
            for sym in all_shape_symbols:
                if sym in solver_json["state_variables"]:
                    iv_expr = shape.get_initial_value(sym.replace(Config().differential_order_symbol, "'"))
                    solver_json["initial_values"][sym] = str(iv_expr)

                    # validate output for numerical problems
                    for var in iv_expr.atoms():
                        _check_numerical_issue(var)

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
                    sympy_expr = sympy.parsing.sympy_parser.parse_expr(param_expr, global_dict=Shape._sympy_globals)

                    # validate output for numerical problems
                    for var in sympy_expr.atoms():
                        _check_numerical_issue(var)

                    # convert to numeric value
                    sympy_expr = sympy_expr.n()

                    # validate output for numerical problems
                    for var in sympy_expr.atoms():
                        _check_numerical_issue(var)

                    solver_json["parameters"][param_name] = str(sympy_expr)


    #
    #   convert expressions from sympy to string
    #

    if type(preserve_expressions) is bool:
        if preserve_expressions:
            # grab all first-order variables
            preserve_expressions = _get_all_first_order_variables(indict)
        else:
            preserve_expressions = []
    elif isinstance(preserve_expressions, Iterable):
        # check that all variables for which preserve_expression was requested were defined as first-order ODE
        first_order_vars = _get_all_first_order_variables(indict)
        for preserve_expressions_var in preserve_expressions:
            if not preserve_expressions_var in first_order_vars:
                raise MalformedInputException("Requested to preserve expression of variable \"" + preserve_expressions_var + "\", but it was not defined as a first-order ODE")
    else:
        raise MalformedInputException("``preserve_expressions`` parameter should be either a boolean or a list of strings corresponding to variable names")

    for solver_json in solvers_json:
        if "update_expressions" in solver_json.keys():
            for sym, expr in solver_json["update_expressions"].items():
                solver_json["update_expressions"][sym] = str(expr)

                if preserve_expressions and sym in preserve_expressions:
                    if "analytic" in solver_json["solver"]:
                        logging.warning("Not preserving expression for variable \"" + sym + "\" as it is solved by propagator solver")
                        continue

                    logging.info("Preserving expression for variable \"" + sym + "\"")
                    var_def_str = _find_variable_definition(indict, sym, order=1)
                    assert var_def_str is not None
                    solver_json["update_expressions"][sym] = var_def_str.replace("'", Config().differential_order_symbol)

        if "propagators" in solver_json.keys():
            for sym, expr in solver_json["propagators"].items():
                solver_json["propagators"][sym] = str(expr)

    logging.info("In ode-toolbox: returning outdict = ")
    logging.info(json.dumps(solvers_json, indent=4, sort_keys=True))

    return solvers_json, shape_sys, shapes


def _init_logging(log_level: Union[str, int] = logging.WARNING):
    """
    Initialise message logging.

    :param log_level: Sets the logging threshold. Logging messages which are less severe than ``log_level`` will be ignored. Log levels can be provided as an integer or string, for example "INFO" (more messages) or "WARN" (fewer messages). For a list of valid logging levels, see https://docs.python.org/3/library/logging.html#logging-levels
    """
    fmt = '%(levelname)s:%(message)s'
    logging.basicConfig(format=fmt)
    logging.getLogger().setLevel(log_level)


def analysis(indict, disable_stiffness_check: bool = False, disable_analytic_solver: bool = False, preserve_expressions: Union[bool, Iterable[str]] = False, simplify_expression: Optional[str] = None, log_level: Union[str, int] = logging.WARNING) -> List[Dict]:
    r"""
    The main entry point of the ODE-toolbox API.

    :param indict: Input dictionary for the analysis. For details, see https://ode-toolbox.readthedocs.io/en/master/#input
    :param disable_stiffness_check: Whether to perform stiffness checking.
    :param disable_analytic_solver: Set to True to return numerical solver recommendations, and no propagators, even for ODEs that are analytically tractable.
    :param preserve_expressions: Set to True, or a list of strings corresponding to individual variable names, to disable internal rewriting of expressions, and return same output as input expression where possible. Only applies to variables specified as first-order differential equations.
    :param simplify_expression: For all expressions ``expr`` that are rewritten internally: the contents of this parameter string are evaluated with ``eval()`` in Python to obtain the final output expression. Override for custom expression simplification steps. Example: ``"sympy.logcombine(sympy.powsimp(sympy.expand(expr)))"``.
    :param log_level: Sets the logging threshold. Logging messages which are less severe than ``log_level`` will be ignored. Log levels can be provided as an integer or string, for example "INFO" (more messages) or "WARN" (fewer messages). For a list of valid logging levels, see https://docs.python.org/3/library/logging.html#logging-levels

    :return: The result of the analysis. For details, see https://ode-toolbox.readthedocs.io/en/latest/index.html#output
    """
    d, _, _ = _analysis(indict,
                        disable_stiffness_check=disable_stiffness_check,
                        disable_analytic_solver=disable_analytic_solver,
                        preserve_expressions=preserve_expressions,
                        simplify_expression=simplify_expression,
                        log_level=log_level)
    return d
