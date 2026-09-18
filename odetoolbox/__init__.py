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
import os
import sys
import sympy
from sympy.core.expr import Expr as SympyExpr
import tempfile

from .config import Config
# (PR107) removed _find_in_matrix logic helper function
from .sympy_helpers import _check_numerical_issue, _check_forbidden_name, _is_zero, _is_sympy_type, SympyPrinter, _sympy_parse_real
from .system_of_shapes import SystemOfShapes
from .shapes import MalformedInputException, Shape
from .expression_optimisation import (
    _apply_cse_to_solver_blocks,
    _serialize_cse_region_blocks,
    _find_non_json_serializable)


try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError as ie:
    logging.getLogger(__name__).warning(
        "PyGSL is not available. The stiffness test will be skipped.")
    logging.getLogger(__name__).warning("Error when importing: " + str(ie))
    PYGSL_AVAILABLE = False

if PYGSL_AVAILABLE:
    # stability profiles of the heavy computational heavy eqs.
    from .stiffness import StiffnessTester

try:
    logging.getLogger("graphviz").setLevel(logging.ERROR)
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
    logging.getLogger(__name__).debug(
        "Finding analytically solvable equations...")

    # dependency edges of the ODE. building a graph showing which variables
    # influence eachother
    dependency_edges = shape_sys.get_dependency_edges()

    if PLOT_DEPENDENCY_GRAPH:

        node_is_analytically_solvable = {sym: False for sym in list(
            shape_sys.x_)}      # checking for linear coefficients

        # this function creates a dict mapping to each vairable to true
        # (solvable) or false (not solvable / non-linear)
        DependencyGraphPlotter.plot_graph(
            shapes,
            dependency_edges,
            node_is_analytically_solvable,
            fn="/tmp/ode_dependency_graph.dot")

    node_is_analytically_solvable = shape_sys.get_lin_cc_symbols(
        dependency_edges, parameters=parameters)

    if PLOT_DEPENDENCY_GRAPH:
        DependencyGraphPlotter.plot_graph(
            shapes,
            dependency_edges,
            node_is_analytically_solvable,
            fn="/tmp/ode_dependency_graph_analytically_solvable_before_propagated.dot")

    # cannot analytically solve inhomogeneous, order > 1 shapes
    for i in range(len(shape_sys.x_)):
        if not _is_zero(shape_sys.b_[i]) and shape_sys.shape_order_from_system_matrix(
                i) > 1 and shape_sys.x_[i] in shape_sys.get_connected_symbols(i):
            node_is_analytically_solvable[shape_sys.x_[i]] = False

    # (#PR107) removes the logic by _find_in_matrix for dependencies of neighbour ODE terms

    # propagating the judgement, if variable a depends on variable b &&
    # variable b (unsolvable) therefore a is unsolvable
    node_is_analytically_solvable = shape_sys.propagate_lin_cc_judgements(
        node_is_analytically_solvable, dependency_edges)
    if PLOT_DEPENDENCY_GRAPH:
        DependencyGraphPlotter.plot_graph(
            shapes,
            dependency_edges,
            node_is_analytically_solvable,
            fn="/tmp/ode_dependency_graph_analytically_solvable.dot")

    return dependency_edges, node_is_analytically_solvable


def _read_global_config(indict):
    r"""
    Process global configuration options.
    """
    logging.getLogger(__name__).debug("Processing global options...")
    if "options" in indict.keys():
        for key, value in indict["options"].items():
            assert key in Config.config.keys(
            ), "Unknown key specified in global options dictionary: \"" + str(key) + "\""
            Config.config[key] = value


def _from_json_to_shapes(
        indict, parameters=None) -> Tuple[List[Shape], Dict[sympy.Symbol, str]]:
    r"""
    Process the indct (json) input, construct Shape instances.

    :param indict: ODE-toolbox input dictionary.
    """

    logging.getLogger(__name__).debug("Processing input...")

    # first run for grabbing all the variable names. Coefficients might be
    # incorrect.
    all_variable_symbols = []
    all_parameter_symbols = set()
    all_variable_symbols_ = set()

    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(shape_json, parameters=parameters)
        all_variable_symbols.extend(shape.get_state_variables())
        all_variable_symbols_.update(shape.get_state_variables(
            derivative_symbol=Config().differential_order_symbol))
        all_parameter_symbols.update(
            set(shape.reconstitute_expr().free_symbols))
    # building master list of every symbol used in nestml
    all_parameter_symbols -= all_variable_symbols_
    del all_variable_symbols_
    assert all([_is_sympy_type(sym) for sym in all_variable_symbols])

    # validate input for forbidden names prevent collisions
    for var in set(all_variable_symbols) | all_parameter_symbols:
        _check_forbidden_name(var)
        assert var.is_real

    # validate parameters
    for param in all_parameter_symbols:          # symbol flagged as param but doesnt have a starting num
        if parameters is None:
            parameters = dict()

        assert isinstance(param, SympyExpr)
        if param not in parameters.keys():
            # this parameter was used in an ODE, but not explicitly numerically
            # specified
            logging.getLogger(__name__).info(
                "No numerical value specified for parameter \"" + str(param) + "\"")    # INFO level because this is OK!
            parameters[param] = None

    # second run with the now-known list of variable symbols
    shapes = []
    # precise definitions instiations with correct boundary terms
    for shape_json in indict["dynamics"]:
        shape = Shape.from_json(
            shape_json,
            all_variable_symbols=all_variable_symbols,
            parameters=parameters)
        shapes.append(shape)

    return shapes, parameters


def _find_variable_definition(indict, name: str, order: int) -> Optional[str]:
    r"""Find the definition (as a string in the input dictionary) of variable named ``name`` with order ``order``, and return it as a string.
    Return None if a definition by that name and order could not be found."""
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
    r"""Return a list of variable names,
    containing those variables that were defined as a first-order ordinary differential equation in the input."""
    variable_names = []

    # iterates through the models dynamics block.
    for dyn in indict["dynamics"]:
        if "expression" in dyn.keys():
            # checking the objects stored under expression block
            exprs = [dyn["expression"]]
        elif "expressions" in dyn.keys():
            exprs = dyn["expressions"]

        for expr in exprs:
            # parsing the ode for (name(V''), order(2) then the mathematical
            # expression string
            name, order, rhs = Shape._parse_defining_expression(expr)
            if order == 1:
                variable_names.append(name)       # filters for first order

    return variable_names


def symbol_appears_in_any_expr(param_name, solver_json) -> bool:
    if "update_expressions" in solver_json.keys():
        for sym, expr in solver_json["update_expressions"].items():
            if param_name in [str(sym) for sym in list(expr.atoms())]:
                return True

    if "propagators" in solver_json.keys():
        for sym, expr in solver_json["propagators"].items():
            if param_name in [str(sym) for sym in list(expr.atoms())]:
                return True

    if "conditions" in solver_json.keys():
        for conditional_solver_json in solver_json["conditions"].values():
            if "update_expressions" in conditional_solver_json.keys():
                for sym, expr in conditional_solver_json["update_expressions"].items(
                ):
                    if param_name in [str(sym) for sym in list(expr.atoms())]:
                        return True

            if "propagators" in conditional_solver_json.keys():
                for sym, expr in conditional_solver_json["propagators"].items(
                ):
                    if param_name in [str(sym) for sym in list(expr.atoms())]:
                        return True

    return False


def _analysis(indict,
              disable_stiffness_check: bool = False,
              disable_analytic_solver: bool = False,
              disable_singularity_detection: bool = False,
              disable_singularity_mitigation: bool = False,
              use_alternative_expM: bool = False,
              preserve_expressions: Union[bool,
                                          Iterable[str]] = False,
              disable_cse: bool = False,
              log_level: Union[str,
                               int] = logging.WARNING) -> Tuple[List[Dict],
                                                                SystemOfShapes,
                                                                List[Shape]]:
    """
    Like analysis(), but additionally returns ``shape_sys`` and ``shapes``.

    For internal use only. External code won't be relying on this but makes it a good area for debugging.
    """

    # import sys;sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

    _init_logging(log_level)

    logging.getLogger(__name__).info("Analysing input:")
    logging.getLogger(__name__).info(
        json.dumps(indict, indent=4, sort_keys=True))

    if "dynamics" not in indict:
        logging.getLogger(__name__).info(
            "Warning: empty input (no dynamical equations found); returning empty output")
        return [], SystemOfShapes.from_shapes([]), []

    _read_global_config(indict)

    # copy parameters from the input and make sure keys are of type
    # sympy.Symbol
    parameters = None
    if "parameters" in indict.keys():
        parameters = {}
        for k, v in indict["parameters"].items():
            if type(k) is str:
                parameters[sympy.Symbol(k, real=True)] = v
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
            logging.getLogger(__name__).error("For symbol " + str(shape.symbol) + ": higher-order inhomogeneous ODEs are not supported")
            sys.exit(1)

    shape_sys = SystemOfShapes.from_shapes(shapes, parameters=parameters)
    _, node_is_analytically_solvable = _find_analytically_solvable_equations(
        shape_sys, shapes, parameters=parameters)

    logging.getLogger(__name__).info(
        "System of equations (with dx/dt = Ax + b + c):")
    logging.getLogger(__name__).info("x = " + str(shape_sys.x_))
    logging.getLogger(__name__).info("A = " + repr(shape_sys.A_))
    logging.getLogger(__name__).info("b = " + str(shape_sys.b_))
    logging.getLogger(__name__).info("c = " + str(shape_sys.c_))

    #
    #   generate analytical solutions (propagators) where possible
    #

    solvers_json = []
    if disable_analytic_solver:
        analytic_syms = []
    else:
        analytic_syms = [
            node_sym for node_sym,
            _node_is_analytically_solvable in node_is_analytically_solvable.items() if _node_is_analytically_solvable]

    analytic_solver_json = None
    if analytic_syms:
        logging.getLogger(__name__).info("Generating propagators for the following symbols: " + ", ".join([str(k) for k in analytic_syms]))
        sub_sys = shape_sys.get_sub_system(analytic_syms)

        # analytical subsystem chosen calling generate propagrator solver
        analytic_solver_json = sub_sys.generate_propagator_solver(
            disable_singularity_detection=disable_singularity_detection,
            disable_singularity_mitigation=disable_singularity_mitigation,
            use_alternative_expM=use_alternative_expM)

        analytic_solver_json["solver"] = "analytical"
        solvers_json.append(analytic_solver_json)

    #
    #   generate numerical solvers for the remainder
    #

    if len(analytic_syms) < len(shape_sys.x_):
        numeric_syms = list(set(shape_sys.x_) - set(analytic_syms))
        logging.getLogger(__name__).info(
            "Generating numerical solver for the following symbols: " + ", ".join([str(sym) for sym in numeric_syms]))
        sub_sys = shape_sys.get_sub_system(numeric_syms)
        solver_json = sub_sys.generate_numeric_solver(
            state_variables=shape_sys.x_)
        # will be appended to if stiffness testing is used
        solver_json["solver"] = "numeric"
        if not disable_stiffness_check:
            if not PYGSL_AVAILABLE:
                raise Exception(
                    "Stiffness test requested, but PyGSL not available")

            logging.getLogger(__name__).info("Performing stiffness test...")
            kwargs = {}   # type: Dict[str, Any]
            if "options" in indict.keys(
            ) and "random_seed" in indict["options"].keys():
                random_seed = int(indict["options"]["random_seed"])
                assert random_seed >= 0, "Random seed needs to be a non-negative integer"
                kwargs["random_seed"] = random_seed
            if "parameters" in indict.keys():
                kwargs["parameters"] = indict["parameters"]
            if "stimuli" in indict.keys():
                kwargs["stimuli"] = indict["stimuli"]
            for key in [
                "sim_time",
                "max_step_size",
                "integration_accuracy_abs",
                    "integration_accuracy_rel"]:
                if "options" in indict.keys() and key in Config().keys():
                    kwargs[key] = float(Config()[key])
            if analytic_solver_json is not None:
                kwargs["analytic_solver_dict"] = analytic_solver_json
            tester = StiffnessTester(sub_sys, shapes, **kwargs)
            solver_type = tester.check_stiffness()

            if solver_type is not None:
                solver_json["solver"] += "-" + solver_type
                logging.getLogger(__name__).info(solver_type + " scheme")
        solvers_json.append(solver_json)

    #
    # copy the initial values from the input to the output for convenience; convert to numeric values
    #

    for solver_json in solvers_json:
        solver_json["initial_values"] = {}
        for shape in shapes:
            all_shape_symbols = [str(sympy.Symbol(str(shape.symbol) + Config().differential_order_symbol * i, real=True)) for i in range(shape.order)]
            for sym in all_shape_symbols:
                if sym in solver_json["state_variables"]:
                    iv_expr = shape.get_initial_value(sym.replace(
                        Config().differential_order_symbol, "'"))
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
                # only make parameters appear in a solver if they are actually
                # used there
                if symbol_appears_in_any_expr(sym, solver_json):
                    sympy_expr = _sympy_parse_real(
                        param_expr, global_dict=Shape._sympy_globals)

                    # validate output for numerical problems -- note that we
                    # skip checking for infinity here as some parameters (like
                    # "V_max") could be legitimately defined as infinity
                    for var in sympy_expr.atoms():
                        _check_numerical_issue(var, check_infty=False)

                    # convert to numeric value
                    sympy_expr = sympy_expr.n()

                    # validate output for numerical problems -- note that we
                    # skip checking for infinity here as some parameters (like
                    # "V_max") could be legitimately defined as infinity
                    for var in sympy_expr.atoms():
                        _check_numerical_issue(var, check_infty=False)

                    # appending parameter discovery to solver json
                    solver_json["parameters"][param_name] = str(sympy_expr)

    #
    # perform cse after parameter discovery whilst expressions are Sympy objects
    #

    if disable_cse: 
        logging.getLogger(__name__).debug("CSE disabled")
    else: 
        logging.getLogger(__name__).debug("Applying CSE to %d solver block(s): %s", len(solvers_json), ", ".join(solver.get("solver","unknown") for solver in solvers_json))

        solvers_json = (_apply_cse_to_solver_blocks(solvers_json))  # pass solver dict as a list into blocks

    #
    #   convert expressions from sympy to string
    #
    #

    if type(preserve_expressions) is bool:
        if preserve_expressions:
            if disable_cse:
                logging.getLogger(__name__).debug("CSE disabled")
            else: 
                raise Exception("If cse is enabled then then this requires setting ``preserve_expressions`` to False for numerical expressions!")

            # grab all first-order variables
            preserve_expressions = _get_all_first_order_variables(indict)
        else:
            preserve_expressions = []
    elif isinstance(preserve_expressions, Iterable):
        # check that all variables for which preserve_expression was requested
        # were defined as first-order ODE
        first_order_vars = _get_all_first_order_variables(indict)
        for preserve_expressions_var in preserve_expressions:
            if preserve_expressions_var not in first_order_vars:
                raise MalformedInputException("Requested to preserve expression of variable \"" + preserve_expressions_var + "\", but it was not defined as a first-order ODE")
    else:
        raise MalformedInputException(
            "``preserve_expressions`` parameter should be either a boolean or a list of strings corresponding to variable names")

    for solver_json in solvers_json:

        #
        # conversion of update_blocks
        #

        if "update_expressions" in solver_json.keys():
            for sym, expr in solver_json["update_expressions"].items():
                solver_json["update_expressions"][sym] = str(expr)

                if preserve_expressions and sym in preserve_expressions:
                    if "analytic" in solver_json["solver"]:
                        logging.getLogger(__name__).warning("Not preserving expression for variable \"" + sym + "\" as it is solved by propagator solver")
                        continue

                    logging.getLogger(__name__).info(
                        "Preserving expression for variable \"" + sym + "\"")
                    var_def_str = _find_variable_definition(
                        indict, sym, order=1)
                    assert var_def_str is not None
                    solver_json["update_expressions"][sym] = var_def_str.replace(
                        "'", Config().differential_order_symbol)

        #
        # conversion of propagators
        #
        if "propagators" in solver_json.keys():
            for sym, expr in solver_json["propagators"].items():
                solver_json["propagators"][sym] = str(expr)

        #
        # conversion of conditional analytical branches
        #
        if "conditions" in solver_json.keys():
            for cond, cond_solver in solver_json["conditions"].items():

                # serialize singularity condition.
                _serialize_cse_region_blocks(cond_solver)

                if "update_expressions" in cond_solver:
                    for sym, expr in cond_solver["update_expressions"].items():
                        cond_solver["update_expressions"][sym] = str(expr)

                        if preserve_expressions and sym in preserve_expressions:
                            if "analytic" in solver_json["solver"]:
                                logging.getLogger(__name__).warning(
                                    "Not preserving expression for variable \"" + sym + "\" as it is solved by propagator solver")
                                continue

                            logging.getLogger(__name__).info(
                                "Preserving expression for variable \"" + sym + "\"")
                            var_def_str = _find_variable_definition(
                                indict, sym, order=1)
                            assert var_def_str is not None
                            cond_solver["update_expressions"][sym] = var_def_str.replace(
                                "'", Config().differential_order_symbol)

                if "propagators" in cond_solver:
                    for sym, expr in cond_solver["propagators"].items():
                        cond_solver["propagators"][sym] = str(expr)

                # serialisation belonging to conditional branches
                if "cse" in solver_json:
                    # CSE for conditional branches
                    _serialize_cse_region_blocks(cond_solver)

        # serialisation cse belonging to this solver.
        if "cse" in solver_json:
            _serialize_cse_region_blocks(
                solver_json)     # CSE for top-level solvers

    logging.getLogger(__name__).info("Final output result:")
    logging.getLogger(__name__).info(
        json.dumps(
            solvers_json,
            indent=4,
            sort_keys=True))

    problems = _find_non_json_serializable(solver_json)
    for path, type_name, value in problems:      # non serialized json bug
        logging.getLogger(__name__).error("non-json value %s -> %s: %s",
                                          path, type_name, value)

    assert not problems, ("Non-JSON safe objects remain in solver_json")

    # output log the json file
    json.dumps(solvers_json, indent=4, sort_keys=True)     # default hides bugs
    return solvers_json, shape_sys, shapes


def _init_logging(log_level: Union[str, int] = logging.WARNING):
    """
    Initialise message logging.
    :param log_level: Sets the logging threshold. Logging messages which are less severe than ``log_level`` will be ignored. Log levels can be provided as an integer or string, for example "INFO" (more messages) or "WARN" (fewer messages). For a list of valid logging levels, see https://docs.python.org/3/library/logging.html#logging-levels
    """
    fmt = "[ODE-toolbox] %(levelname)s:%(message)s"
    logging.basicConfig(format=fmt)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)


def analysis(indict,
             disable_stiffness_check: bool = False,
             disable_analytic_solver: bool = False,
             disable_singularity_detection: bool = False,
             disable_singularity_mitigation: bool = False,
             use_alternative_expM: bool = False,
             disable_cse: bool = False,
             preserve_expressions: Union[bool,
                                         Iterable[str]] = False,
             log_level: Union[str,
                              int] = logging.WARNING) -> List[Dict]:
    r"""
    The main entry point of the ODE-toolbox API.

    :param indict: Input dictionary for the analysis. For details, see https://ode-toolbox.readthedocs.io/en/main/#input
    :param disable_stiffness_check: Whether to perform stiffness checking.
    :param disable_analytic_solver: Set to True to return numerical solver recommendations, and no propagators, even for ODEs that are analytically tractable.
    :param disable_singularity_detection: Set to True to disable detection of conditions under which numerical singularities (division by zero) could occur in the generated analytic solver. This can be useful for analytic solvers containing a large amount of conditions, which could take a long time to compute. If True, no warning messages or mitigations will be returned in case of potential singularities.
    :param disable_singularity_mitigation: (Only useful if ``disable_singularity_detection`` is False.) Set to True to disable mitigation of conditions under which numerical singularities (division by zero) could occur in the generated analytic solver. This can be useful for analytic solvers containing a large amount of conditions, which could take a long time to compute. If True, at most one analytic solver will be returned, in which numerical singularities could occur.
    :param use_alternative_expM: If :python:`False`, use the sympy function ``sympy.exp`` to compute the matrix exponential. If :python:`True`, use an alternative function (see :py:func:`odetoolbox.sympy_helpers.expMt` for details). This can be useful as calls to ``sympy.exp`` can sometimes take a very large amount of time.
    :param preserve_expressions: Set to True, or a list of strings corresponding to individual variable names, to disable internal rewriting of expressions, and return same output as input expression where possible. Only applies to variables specified as first-order differential equations.
    :param log_level: Sets the logging threshold. Logging messages which are less severe than ``log_level`` will be ignored. Log levels can be provided as an integer or string, for example "INFO" (more messages) or "WARN" (fewer messages). For a list of valid logging levels, see https://docs.python.org/3/library/logging.html#logging-levels
    :param disable_cse: Boolean flag set to False. If enabled it will perform sub-expression elimination on update_expression, propagators and singularity conditions of the generated .cpp nestml file.

    :return: The result of the analysis. For details, see https://ode-toolbox.readthedocs.io/en/latest/index.html#output
    """
    d, _, _ = _analysis(indict,
                        disable_stiffness_check=disable_stiffness_check,
                        disable_analytic_solver=disable_analytic_solver,
                        disable_singularity_detection=disable_singularity_detection,
                        disable_singularity_mitigation=disable_singularity_mitigation,
                        use_alternative_expM=use_alternative_expM,
                        preserve_expressions=preserve_expressions,
                        disable_cse=disable_cse,
                        log_level=log_level)
    return d
