#
# expression_optimisation.py
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

"""
Expression optimisation helper functions used by the ODE-toolbox analysis pipeline.
This module contains runtime common subexpression elimination (CSE) functionality and test specific validations.
"""

import logging
import sympy

OP_WEIGHTS = {  # define individual weights of expressions
    sympy.Add: 1.0,
    sympy.Mul: 1.0,
    sympy.Pow: 4.0,
    sympy.exp: 8.0,
    sympy.log: 8.0,
    sympy.sin: 8.0,
    sympy.cos: 8.0,
}

TEMPORARY_OVERHEAD = 1.0  # cost of creating of a tmp variable
# cost of optimal expr must be greater than to be accepted
MIN_TEMPORARY_NET_BENEFIT = 3.0


def common_subexpression_elimination(
        expressions,
        symbol_prefix="__ode_cse_tmp__"):
    """
    custom wrapper to perform common subexpression elimination across mapping of a
    named sympy expression while preserving expression ordering.
    """

    if not expressions:  # if expressions are empty
        return [], {}

    # expression names in specific order
    expressions_names = list(expressions.keys())

    # sympy accepts ordered list of mathematical expressions in same order as
    # expression_names
    expressions_values = [expressions[name] for name in expressions_names]

    # check for sympy objects
    if not all(isinstance(expression, sympy.Basic)
               for expression in expressions_values):
        raise TypeError(
            "CSE expects Sympy objects. String serialisation has not occured. ")

    # infinite generator for the temp local variables to hold isolated math
    # subexpressions
    temporary_symbols = sympy.numbered_symbols(symbol_prefix)

    # perform cse
    replacements, reduced_values = sympy.cse(
        expressions_values,
        symbols=temporary_symbols,
        optimizations=None,
        # forces ordering deterministically (e.g., A relies on B)
        order="canonical"
    )

    # fuse cse expressions with values, maintain order of eq.
    reduced_expressions = dict(zip(expressions_names, reduced_values))

    # replacement is returned as tuples where each tupe contains the temp
    # var_name with the maths block
    return replacements, reduced_expressions


def weighted_expression_cost(expr):
    """
    estimate expression cost using operation-specific weights, where adding expr has low weight etc. and sin(x) has high weight
    this is a heuristic machine-aware cost, not a prediciton of exact CPU cycles or instruction counts.
    """

    if not isinstance(expr, sympy.Basic):
        raise TypeError(
            f"weighted expression costs expect a SymPy expression. Received : {
                type(expr).__name__}")

    if expr.is_Atom:  # individual numbers/ symbols doesnt require any calculation
        return 0.0

    # Look up the cost of the current operation (defualt 1.0)
    # expr.func tells us if it's an Add, Mul, Pow, sin, exp, etc with an
    # assigned weight
    weight = OP_WEIGHTS.get(expr.func, 1.0)

    if expr.func in (
            sympy.Add,
            sympy.Mul):  # Handle chaining for additions/multiplications (e.g., x + y + z has 2 operations)

        # calculates how many binary ops occur when you add/multiply together
        own_cost = max(0, len(expr.args) - 1) * weight

    else:
        own_cost = weight  # if it's a singular operation assign weight

    # calculate cost and every expression inside them
    child_cost = sum(weighted_expression_cost(arg) for arg in expr.args)

    return own_cost + child_cost


def count_symbol_uses(symbol, remaining_replacements, reduced_expressions):
    """
    count how many times a cse temporary symbol is referred by later replacements and later final reduced expressions
    """
    uses = 0

    for _, expr in remaining_replacements:
        uses += expr.count(symbol)

    for expr in reduced_expressions.values():
        uses += expr.count(symbol)

    return uses


def estimate_total_cost(expressions):
    """
    Return weight estimate machine cost of an iterable of expressions
    """
    return sum(weighted_expression_cost(expr) for expr in expressions)


def _run_profitable_cse(
        expressions,
        symbol_prefix,
        solver_name="unknown",
        region_name="unknown"):
    """
    Run SymPy CSE, remove low-benefit candidate temporaries,
    and retain the final transformation only when it still
    reduces estimated expression cost.
    """

    logger = logging.getLogger(__name__)

    if not expressions:
        return [], expressions

    # run cse expression
    replacements, reduced = common_subexpression_elimination(
        expressions, symbol_prefix=symbol_prefix)

    if not replacements:
        logger.debug(
            "[CSE] solver=%s region=%s: "
            "no common subexpressions found", solver_name, region_name,)
        return [], expressions

    original_count = len(replacements)  # original count of cse
    replacements, reduced = filter_cse_replacements(
        replacements, reduced)  # prune out unprofitable cse

    if not replacements:
        logger.debug(
            "[CSE] solver=%s region=%s: "
            "all %d candidate temporaries were rejected",
            solver_name,
            region_name,
            original_count)
        return [], expressions

    # calculate the cost of expressions before/after cse applied
    before_cost = estimate_total_cost(expressions.values())
    after_cost = (
        estimate_total_cost(
            expr for _,
            expr in replacements) + estimate_total_cost(reduced.values()) + len(replacements) * TEMPORARY_OVERHEAD)

    if after_cost >= before_cost:

        logger.debug(  # inline rejection
            "[CSE] solver=%s region=%s: "
            "final filtered CSE rejected "
            "(before=%.2f after=%.2f)", solver_name, region_name, before_cost, after_cost)

        return [], expressions

    logger.debug(  # accepted with cse solver statistics
        "[CSE] solver=%s region=%s: ACCEPTED "
        "(candidates=%d kept=%d removed=%d "
        "before=%.2f after=%.2f reduction=%.2f%%)",
        solver_name, region_name, original_count, len(replacements),
        original_count - len(replacements), before_cost, after_cost,
        ((before_cost - after_cost) / before_cost * 100.0))

    return replacements, reduced


def filter_cse_replacements(
        replacements,
        reduced_expressions,
        min_net_benefit=MIN_TEMPORARY_NET_BENEFIT):
    """
    Rejected temporaries are inlined into later replacements and into
    the final reduced expressions so that dependency ordering and
    mathematical equivalence are preserved.
    """

    logger = logging.getLogger(__name__)
    kept_replacements = []
    # Maps rejected temporary symbols back to their expressions.
    substitutions = {}

    for index, (symbol, expression) in enumerate(
            replacements):  # evaluating each temporary

        # Inline any earlier temporary that was rejected.
        expression = expression.xreplace(substitutions)

        # Remaining expressions must also be viewed after existing
        # rejected substitutions have been inlined.
        # ensures we dont reject some tmp variables that are used later in
        # expressions
        remaining_replacements = [(later_symbol, later_expr.xreplace(
            substitutions)) for later_symbol, later_expr in replacements[index + 1:]]

        current_reduced = {
            name: expr.xreplace(substitutions)
            for name, expr in reduced_expressions.items()}

        # count number of times a specific variable is used in the rest of the
        # script
        uses = count_symbol_uses(
            symbol,
            remaining_replacements,
            current_reduced)
        expression_cost = weighted_expression_cost(
            expression)  # weighted expression cost
        # calculate gross beneft of using tmp with expression cost and uses
        gross_benefit = (expression_cost * max(0, uses - 1))
        # tmp overhead cost for allocating memory for the new tmp variable.
        net_benefit = (gross_benefit - TEMPORARY_OVERHEAD)

        if net_benefit >= min_net_benefit:  # greater than 3

            kept_replacements.append((symbol, expression))

            logger.debug(
                "[CSE TEMP] symbol=%s cost=%.2f uses=%d gross_benefit=%.2f net_benefit=%.2f decision=KEEP expr=%s",
                symbol,
                expression_cost,
                uses,
                gross_benefit,
                net_benefit,
                expression)
        else:

            # record the temporary even if rejected, as it might get used later
            # in script
            substitutions[symbol] = expression

            logger.debug(
                "[CSE TEMP] symbol=%s cost=%.2f uses=%d gross_benefit=%.2f net_benefit=%.2f decision=INLINE expr=%s",
                symbol,
                expression_cost,
                uses,
                gross_benefit,
                net_benefit,
                expression)

    # Inline every rejected temporary variable into surviving replacement
    # expressions so we have template of accepted/rejections
    final_replacements = [(symbol, expression.xreplace(substitutions))
                          for symbol, expression in kept_replacements]

    # final output for mathematical expressions
    final_reduced = {
        name: expression.xreplace(substitutions) for name,
        expression in reduced_expressions.items()}
    return final_replacements, final_reduced


def _contains_nonfinite_expression(expressions):
    """
    Check for symbolic infinities, this looks at an equation and determines that it will always evaluate to infinity or divide by 0,
    regardless of the numerical values you pass. This is secondary sanity check, as all symbolic infinities should theortically be filtered out
    by the singularity conditions.
    """
    invalid_values = (
        sympy.zoo,  # complex infinity
        sympy.oo,  # infinity
        -sympy.oo,  # negative infinity
        sympy.nan  # NaN
    )

    for expression in expressions:
        if hasattr(expression, "has"):
            for invalid in invalid_values:
                if expression.has(invalid):
                    return True  # if the value contains these invalid values return true

    return False


def _contains_internal_control_flow(expressions):
    """
    If a sympy.Piecewise object is hidden inside an expression, it introduces hidden branching logic. If CSE blindly pulls an equation out from inside a
    Piecewise condition and places it at the global scope (when it has a local conditional specifications), it forces the CPU to compute it all the time.
    This ruins your conditional optimization and can lead to runtime NaN crashes or division-by-zero errors. This acts a secondary safety check before cse.
    """

    # returns True/False for any sympy Piecewise is present
    return any(
        isinstance(
            expression,
            sympy.Basic) and expression.has(
            sympy.Piecewise) for expression in expressions)


def _apply_cse_to_expression_region(
        region,
        symbol_prefix,
        solver_name="unknown"):
    """
    Apply CSE independently inside one execution region.

    The condition controlling execution region is not modified, and therefore this function should never
    receive multiple singularity branches as this function does not hold logic for interpreting these singularities.

    'propagator' 'update_expressions' are also handled differently as they are executed in different contexts down stream.
    """

    result = dict(region)
    cse_data = {}  # use a dict instead of a list [] to avoid KeyError downstream

    # collect all expressions safely into a list for the non-finite check
    all_math_expressions = []

    if region.get(
            "propagators"):  # collect update_expressions values inside all_math_expressions
        all_math_expressions.extend(region["propagators"].values())

    if region.get(
            "update_expressions"):  # collect propagator values inside all_math_expressions
        all_math_expressions.extend(region["update_expressions"].values())

    # safety check before running cse in singularity
    if _contains_nonfinite_expression(
            all_math_expressions):  # second sanity check for singularities
        logger = logging.getLogger(__name__)
        logger.debug(
            "Skipping CSE for region %s: non-finite symbolic expression detected",
            symbol_prefix)
        return result

    # safety check before running cse in singularity
    if _contains_internal_control_flow(all_math_expressions):
        logger = logging.getLogger(__name__)
        logger.debug(
            "Skipping CSE for region %s: nested SymPy Piecewise expression detected",
            symbol_prefix)
        return result

    # analytical
    if region.get("propagators"):
        replacements, reduced = _run_profitable_cse(
            region["propagators"], symbol_prefix + "prop_",
            solver_name=solver_name, region_name="propagators")
        if replacements:
            result["propagators"] = reduced
            cse_data["propagators"] = replacements

    # numerical state update expressions
    if region.get("update_expressions"):
        replacements, reduced = _run_profitable_cse(
            region["update_expressions"], symbol_prefix + "update_",
            solver_name=solver_name, region_name="update_expressions")
        if replacements:
            result["update_expressions"] = reduced
            cse_data["update_expressions"] = replacements

    if cse_data:
        result["cse"] = cse_data  # output data

    return result


def apply_cse_to_solver(solver, symbol_prefix="__ode_cse_"):
    """
    Apply CSE to one ODE-toolbox solver dictionary. Singularity branches are treated as independent execution regions that has no variable relationships to eachother. 
    Both solvers are passed to _apply_cse_to_expression_region 
    """

    result = dict(solver)
    solver_name = solver.get("solver", "unknown")

    if "conditions" in solver:  # handle singularity conditions

        result = dict(solver)
        optimised_conditions = {}

        # wrap condition, branch so python understand how to unpack a sub-tuple
        for branch_index, (condition, branch) in enumerate(
                solver["conditions"].items()):

            # distinct condition blocks get their own unique prefix
            branch_prefix = (symbol_prefix + f"cond_{branch_index}_")

            optimised_conditions[condition] = _apply_cse_to_expression_region(
                branch,
                symbol_prefix=branch_prefix,
                solver_name=solver_name)  # apply cse to each independent branch

        result["conditions"] = optimised_conditions

        return result  # return result if we've conducted cse singularity

    # ordinary analytical or numerical solver passing.
    return _apply_cse_to_expression_region(
        solver, symbol_prefix=symbol_prefix, solver_name=solver_name)


def _apply_cse_to_solver_blocks(solver_blocks):
    """
    Apply CSE independently to every solver block produced by ODEtoolbox

    A mixed system can contain both analytical and numerical blocks. Give each block a seperate tmp variable namespace so that CSE
    temporaries cannot leak or collide across solver blocks.
    """

    if not isinstance(solver_blocks, list):
        raise TypeError(
            "_apply_cse_to_solver_blocks() expects a list of solver dictionaries",
            f"received: {
                type(solver_blocks).__name__}")

    result = []
    multiple_blocks = len(solver_blocks) > 1

    for block_index, solver in enumerate(solver_blocks):

        logging.getLogger(__name__).debug(
            "Applying CSE to solver block %d (%s)",
            block_index,
            solver.get("solver", "unknown"))

        if multiple_blocks:
            symbol_prefix = f"__ode_cse_solver_{block_index}_"
        else:
            symbol_prefix = f"__ode_cse_"

        result.append(apply_cse_to_solver(solver, symbol_prefix=symbol_prefix))

    return result


def serialize_replacements(replacements):
    """
    Convert symbolic CSE replacement pairs into string, JSON-safe key-value pairs.
    Transforming a list of tuples [(x, a + b)] into a clean dictionary {"x": "a + b"}.
    """

    result = {}
    for symbol, expr in replacements:
        result[str(symbol)] = str(expr)

    return result

def _serialize_cse_region_blocks(region):
    """
    Serialize the internal cse subexpressions belonging to a single solver and iterates  through through the structure logic blocks 
    (e.g., specific branches or updates) and replaces raw Sympy symbolic data with JSON-strings. 
    """

    # pass if cse wasn't conducted
    if "cse" not in region:
        return

    for expression_region, replacements in (list(region["cse"].items())):

        # call the indivudal serialise functon once within the solver
        region["cse"][expression_region] = serialize_replacements(replacements)

def _find_non_json_serializable(obj, path="root"):
    """
    Recursively inspect an object to isolate elements that cannot be serialized to JSON and flag them to user.
    """

    # Clear terminal conditions for valid JSON types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return []

    # Handle dictionaries safely
    if isinstance(obj, dict):
        problems = []
        for key, value in obj.items():
            if not isinstance(key, (str, int, float, bool, type(None))):
                # Wrapped in a proper 3-element tuple inside the append
                problems.append(
                    (f"{path}.<key>", type(key).__name__, repr(key)))
            problems.extend(_find_non_json_serializable(
                value, path=f"{path}[{key!r}]"))
        return problems

    # Handle sequences
    if isinstance(obj, (list, tuple)):
        problems = []
        for index, value in enumerate(obj):
            problems.extend(_find_non_json_serializable(
                value, path=f"{path}[{index}]"))
        return problems

    # Catch-all for non-serializable objects (like SymPy Symbols)
    # Corrected to a uniform list containing a single 3-element tuple
    return [(path, type(obj).__name__, repr(obj))]

#
#
# cse optimisation helper functions for validation during cse_testing
#
#


def deserialize_cse_replacements(replacements):
    """
    deserialize ordered cse replacement definitions into Sympy expressions. Takes a mapping of temporary symbol names to their serialized string expressions, 
    declares the symbols as real-valued SymPy Symbols, and parses the expressions while maintaining proper scoping dependencies between intermediate terms.
    """

    from .shapes import Shape
    from .sympy_helpers import _sympy_parse_real

    if not replacements:
        return []

    if not isinstance(replacements, dict):
        raise TypeError("expected cse replacements to be dict")

    # define tmp symbols as smypy obj
    temporary_symbols = {
        name: sympy.Symbol(
            name,
            real=True) for name in replacements}

    result = []

    for name, expression in replacements.items():

        parsed_expr = _sympy_parse_real(
            str(expression),
            global_dict=Shape._sympy_globals,
            local_dict=temporary_symbols)  # ensures 'beta' string isnt being treated like a var

        result.append((temporary_symbols[name], parsed_expr))

    return result


def expand_cse_expressions(reduced_expressions, replacements):
    """
    expand one cse-reduced expression back to a normal sympy expression.
    """

    from .shapes import Shape
    from .sympy_helpers import _sympy_parse_real

    # convert serialized replacement into ordered sympy
    replacements = deserialize_cse_replacements(replacements)

    # define tmp symbols as smypy obj
    temporary_symbols = {str(symbol): symbol for symbol, _ in replacements}

    if isinstance(
            reduced_expressions,
            sympy.Basic):  # true for constants and single variables
        expression = reduced_expressions

    else:
        # Custom parse internal function to make sure that all returned symbols
        # have domain Real
        expression = _sympy_parse_real(
            str(reduced_expressions),
            global_dict=Shape._sympy_globals,
            local_dict=temporary_symbols)  # ensures 'beta' string isnt being treated like a var

    #
    # Replacements are in dependency ordered.
    #
    for temporary, replacement in reversed(replacements):

        # substition for the past term
        expression = expression.xreplace({temporary: replacement})

    return expression  # return raw expression


def expand_cse_solver(solver):
    """
    convert an ode-toolbox solver containing serialized CSE definitions into an ordinary expression solver. 
    Does not modify the original solver instance.
    """

    import copy
    result = copy.deepcopy(solver)  # make a scratch deep copy of the solver
    
    # isolate cse tmp translations high level cdict
    cse_substitions = result.get("cse", {})

    for region_name in ("propagators", "update_expressions"):
        if region_name not in result:
            continue

        # pull out keys inside cse dict
        replacements = cse_substitions.get(region_name)

        if not replacements:
            continue

        # swaps out expressions, replacement tmp for full, raw maths ops
        result[region_name] = {
            expression_name: expand_cse_expressions(
                expression,
                replacements) for expression_name,
            expression in result[region_name].items()}

        #
        # conditional analytical sovler branches can carry cse also
        #

    if "conditions" in result:

        # recursion calls itself, inside each block to solver those blocks
        # inside the blocks !
        result["conditions"] = {
            condition: expand_cse_solver(conditional_solver) for condition,
            conditional_solver in result["conditions"].items()}

    #
    # drop the now-expanded CSE block and return the result
    #
    result.pop("cse", None)

    return result


def _has_cse(d):
    """
    This helper function recursively searches for dictionaries containing the key "cse", searching through top-level
    as well as nested structures behind "conditions"
    """

    if not isinstance(d, dict):  # ensure input is a dictionary
        return False

    if any(str(k).lower() == "cse" for k in d.keys()
           ):  # case sensitive top-level search
        return True

    if "conditions" in d:  # if conditions exist, loop through all sub-dicts inside searching for cse
        return any(_has_cse(branch) for branch in d["conditions"].values())
    return False
