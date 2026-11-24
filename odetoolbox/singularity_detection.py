#
# singularity_detection.py
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
from typing import Dict, List, Optional, Set, Union

import logging
import sympy

from .sympy_helpers import SymmetricEq, _sympy_floats_to_ints, symbol_in_expression, _is_zero


class SingularityDetectionException(Exception):
    """
    Thrown in case an error occurs while detecting singularities.
    """
    pass


class SingularityDetection:
    r"""Singularity detection for generated propagator matrix.

    Some ordinary differential equations (ODEs) can be solved analytically: an expression for the solution can be readily derived by algebraic manipulation. This allows us to formulate an "exact integrator", that yields the next state of the system given the current state and the timestep Δt, to floating point (machine) precision [1]_.

    In some cases, an ODE is analytically tractable, but vulnerable to an edge case condition in the generated propagator matrices. Consider the following example: Let the system of ODEs be given by

    .. math::

       y' = A \cdot y

    Then the propagator matrix for a timestep :math:`\Delta t` is

    .. math::

       P = \exp(A \cdot \Delta t)

    which we can use to advance the system

    .. math::

       y(t + \Delta t) = P \cdot y(t)

    If :math:`A` is of the form:

    .. math::

       \begin{bmatrix}
       -a & 0  & 0\\
       1  & -a & 0\\
       0  & 1  & -b
       \end{bmatrix}

    Then the generated propagator matrix contains denominators that include the factor :math:`a - b`. When the parameters are chosen such that :math:`a = b`, a singularity (division by zero fault) occurs. However, the singularity is readily avoided if we assume that :math:`a = b` before generating the propagator, i.e. we start out with the matrix

    .. math::

       \begin{bmatrix}
       -a & 0  & 0\\
       1  & -a & 0\\
       0  & 1  & -a
       \end{bmatrix}

    The resulting propagator contains no singularities.

    This class detects the potential occurrence of such singularities (potential division by zero) in the generated propagator matrix, which occur under certain choices of parameter values. These choices are reported as "conditions" by the ``find_singularities()`` function.


    References
    ----------

    .. [1] Stefan Rotter, Markus Diesmann. Exact digital simulation of time-invariant linear systems with applications to neuronal modeling. Neurobiologie und Biophysik, Institut für Biologie III, Universität Freiburg, Freiburg, Germany Biol. Cybern. 81, 381-402 (1999)
    """

    @staticmethod
    def _is_matrix_defined_under_substitution(A: sympy.Matrix, cond: Union[SymmetricEq, Set[SymmetricEq]]) -> bool:
        r"""
        Function to check if a matrix is defined (i.e. does not contain NaN or infinity) after we perform a given set of substitutions.

        Parameters
        ----------
        A : sympy.Matrix
            input matrix
        cond : Set(SymmetricEq)
            a set with equations, where the left-hand side of each equation is the variable that is to be substituted, and the right-hand side is the expression to put in its place
        """
        for val in sympy.flatten(A):
            if isinstance(val, float) or isinstance(val, int) or isinstance(val, sympy.core.numbers.Number):
                # a number is always defined under substitution
                continue

            expr_sub = val
            if isinstance(cond, set):
                for _cond in cond:
                    expr_sub = expr_sub.subs(_cond.lhs, _cond.rhs)
            else:
                expr_sub = expr_sub.subs(cond.lhs, cond.rhs)

            if symbol_in_expression([sympy.nan, sympy.zoo, sympy.oo], sympy.simplify(expr_sub)):
                return False

        return True

    @staticmethod
    def _filter_valid_conditions(conds: Set[SymmetricEq], A: sympy.Matrix):
        filt_cond = set()
        for cond in conds:
            if SingularityDetection._is_matrix_defined_under_substitution(A, cond):
                filt_cond.add(cond)

        return filt_cond

    @staticmethod
    def _generate_singularity_conditions(P: sympy.Matrix) -> Set[Dict[sympy.core.expr.Expr, sympy.core.expr.Expr]]:
        r"""
        For each element of ``P``, find conditions under which subterms of ``expr`` of the form ``a / b`` equal infinity (in general, when b = 0).
        """
        conditions = set()
        for expr in sympy.flatten(P):
            conds = SingularityDetection._find_singularity_conditions_in_expression(expr)
            conditions = conditions.union(conds)

        return conditions

    @staticmethod
    def _get_singular_values(singularity_set: sympy.Set) -> List[sympy.core.expr.Expr]:
        r"""
        Return the elements of the set returned by ``sympy.calculus.singularities()``.

        If a singular value cannot be decided to be real, for instance :math:`1/g` (which is not real for :math:`g = 0`), sympy returns the unevaluated set ``Intersection({1/g}, Reals)``, which cannot be iterated over. In that case, the elements of the finite set are returned, except for those that are known not to be real. Whether the substitution is valid is checked afterwards (see :py:meth:`_filter_valid_conditions`).
        """
        if isinstance(singularity_set, sympy.FiniteSet):
            return list(singularity_set)

        if isinstance(singularity_set, sympy.Union):
            return [val for subset in singularity_set.args for val in SingularityDetection._get_singular_values(subset)]

        if isinstance(singularity_set, sympy.Intersection) and sympy.Reals in singularity_set.args:
            finite_sets = [subset for subset in singularity_set.args if isinstance(subset, sympy.FiniteSet)]
            other_sets = [subset for subset in singularity_set.args if not isinstance(subset, sympy.FiniteSet) and subset != sympy.Reals]
            if len(finite_sets) == 1 and not other_sets:
                return [val for val in finite_sets[0] if val.is_real is not False]

        # fall back on default iteration (raises an exception if set membership cannot be decided)
        return list(singularity_set)

    @staticmethod
    def _find_singularity_conditions_in_expression(expr: sympy.core.expr.Expr) -> Set[SymmetricEq]:
        r"""Find conditions under which subterms of ``expr`` of the form ``a / b`` equal infinity (in general, when b = 0)."""
        conditions = set()

        sub_expr_conds = set()
        for symbol in expr.free_symbols:
            assert symbol.is_real
            singularity_set = sympy.calculus.singularities(expr, symbol=symbol, domain=sympy.Reals)
            for singular_value in SingularityDetection._get_singular_values(singularity_set):
                singular_value = _sympy_floats_to_ints(singular_value)
                singular_condition = SymmetricEq(symbol, singular_value)
                sub_expr_conds.add(singular_condition)

        conditions = conditions.union(sub_expr_conds)

        return conditions

    @staticmethod
    def find_inhomogeneous_singularities(A: sympy.Matrix, b: sympy.Matrix, Qb: Optional[sympy.Matrix] = None) -> Set[SymmetricEq]:
        r"""Find singularities in the inhomogeneous part of the update equations.

        If the inhomogeneous update term :math:`Qb` (as obtained by the Van Loan method, see :py:meth:`SystemOfShapes._exponentiate_block`) is given, each of its entries is checked. :math:`Qb` contains singularities wherever (a block of) :math:`A` becomes singular for particular parameter values, which need not coincide with a zero on the diagonal of :math:`A`. Moreover, a constant input in one row is propagated through :math:`A` into the :math:`Qb` of downstream rows that have no inhomogeneous term of their own, e.g. for :math:`y' = -ky + I,\; x' = -gx + y`, the update term for :math:`x` contains a factor :math:`1/g`, while :math:`b_x = 0`.

        If :math:`Qb` is not given, or if checking one of its entries fails, the particular solution :math:`-b_i / A_{ii}` is checked instead, for each row :math:`i` with :math:`b_i \neq 0` (or only for the rows whose check failed). This is only exact for rows that are decoupled from the other rows, and is not performed when :math:`Qb` could be checked, because it can yield spurious conditions: for instance, for :math:`x'' = -ab x - (a + b) x' + I`, it yields :math:`a = -b`, for which the system is not singular.

        Parameters
        ----------
        A : sympy.Matrix
            system matrix
        b : sympy.Matrix
            inhomogeneous term
        Qb : Optional[sympy.Matrix]
            inhomogeneous update term (one entry per row of :math:`A`)

        Returns
        -------

        conditions
            a set with equations, where the left-hand side of each equation is the variable that is to be substituted, and the right-hand side is the expression to put in its place
        """
        logging.getLogger(__name__).debug("Checking for singularities (divisions by zero) in the inhomogeneous part of the update equations...")

        try:
            conditions = set()

            if Qb is None:
                rows_to_check_particular_solution = range(A.shape[0])
            else:
                rows_to_check_particular_solution = []
                for row in range(Qb.shape[0]):
                    if _is_zero(Qb[row]):
                        continue

                    try:
                        conditions = conditions.union(SingularityDetection._find_singularity_conditions_in_expression(Qb[row]))
                    except Exception as e:
                        # keep the conditions that were found so far; fall back on checking the particular solution for this row
                        logging.getLogger(__name__).warning("Could not check the inhomogeneous update term for singularities: " + str(e))
                        rows_to_check_particular_solution.append(row)

            for row in rows_to_check_particular_solution:
                if _is_zero(b[row]):
                    # this is not an inhomogeneous ODE
                    continue

                particular_solution = -b[row] / A[row, row]
                particular_solution = sympy.simplify(particular_solution)    # using _custom_update_expr() does not guarantee that an adequate number of cases will be covered

                conditions = conditions.union(SingularityDetection._find_singularity_conditions_in_expression(particular_solution))

            conditions = SingularityDetection._filter_valid_conditions(conditions, A.row_join(b))  # filters out the invalid conditions (invalid means those for which A or b is not defined)
            conditions = SingularityDetection._remove_duplicate_conditions(conditions)
        except Exception as e:
            raise SingularityDetectionException(str(e)) from e

        return conditions

    @staticmethod
    def _conditions_are_equivalent(cond1: SymmetricEq, cond2: SymmetricEq) -> bool:
        r"""Check whether each of the two conditions holds when the other one is substituted into it, for example :math:`g = 1/\tau` and :math:`\tau = 1/g`."""
        try:
            for c1, c2 in [(cond1, cond2), (cond2, cond1)]:
                residual = (c2.lhs - c2.rhs).subs(c1.lhs, c1.rhs)
                if not sympy.simplify(residual) == 0:
                    return False
        except Exception:
            return False

        return True

    @staticmethod
    def _conditions_are_contradictory(cond_set: Set[Union[SymmetricEq, sympy.Ne]]) -> bool:
        r"""Check whether a combination of equalities (``SymmetricEq``) and inequalities (``sympy.Ne``) cannot hold simultaneously, for example :math:`g = 1/\tau \wedge g = 0`, or :math:`k = g \wedge k = 0 \wedge g \neq 0`.

        The equalities are solved jointly; the combination is contradictory if there is no solution, or if each solution violates at least one of the inequalities. If this cannot be decided, the combination is assumed not to be contradictory.

        The check is only performed if all equalities are rational functions of the parameters. For other equalities, ``sympy.solve()`` can return an incomplete set of solutions; for instance, for :math:`\tau = -\sqrt{c} \wedge c = \tau^2`, it returns only :math:`c = \tau = 0`, while all :math:`\tau \leq 0` are solutions.
        """
        equalities = [cond for cond in cond_set if isinstance(cond, SymmetricEq)]
        inequalities = [cond for cond in cond_set if not isinstance(cond, SymmetricEq)]

        if not equalities:
            return False

        try:
            symbols = sorted(set().union(*[eq.free_symbols for eq in equalities]), key=str)

            if not all((eq.lhs - eq.rhs).is_rational_function(*symbols) for eq in equalities):
                return False

            solutions = sympy.solve([eq.lhs - eq.rhs for eq in equalities], symbols, dict=True)

            if not solutions:
                return True

            for solution in solutions:
                if not any(sympy.simplify((ineq.lhs - ineq.rhs).subs(solution)) == 0 for ineq in inequalities):
                    # this solution satisfies all the inequalities
                    return False

            return True

        except Exception as e:
            logging.getLogger(__name__).debug("Could not check whether conditions " + str(cond_set) + " are contradictory: " + str(e))
            return False

    @staticmethod
    def log_conditions(conditions: Set[SymmetricEq], message: str, previously_reported_conditions: Optional[Set[SymmetricEq]] = None) -> None:
        r"""Log a warning listing the given conditions, skipping those that are identical or equivalent to a condition in ``previously_reported_conditions`` (so that the same condition is not reported twice). If no conditions remain, nothing is logged."""
        if previously_reported_conditions is None:
            previously_reported_conditions = set()

        conditions_to_report = []
        for cond in conditions:
            already_reported = False
            for prev_cond in previously_reported_conditions:
                if cond == prev_cond \
                   or SymmetricEq(-cond.lhs, -cond.rhs) == prev_cond \
                   or SingularityDetection._conditions_are_equivalent(cond, prev_cond):
                    already_reported = True
                    break

            if not already_reported:
                conditions_to_report.append(cond)

        if not conditions_to_report:
            return

        # if there is one or more condition under which the solution goes to infinity...
        logging.getLogger(__name__).warning(message)
        logging.getLogger(__name__).warning("List of all conditions that result in a division by zero:")
        for cond in conditions_to_report:
            logging.getLogger(__name__).warning("\t" + str(cond.lhs) + " = " + str(cond.rhs))

    @staticmethod
    def _remove_duplicate_conditions(conditions: Set[SymmetricEq]):
        r"""Remove duplicated conditions. ``conditions`` is already a set of ``SymmetricEq`` (so that ``a == b`` is equivalent to ``b == a``), so duplicates should normally not be possible anyway, but sometimes, in addition to ``a == b``, the condition ``-a == -b`` is present, which is effectively a duplicate. Similarly, the same condition can be found solved for different symbols, such as :math:`g = 1/\tau` and :math:`\tau = 1/g`; in that case, the simplest of the two is kept. This method removes those kinds of duplicates."""
        for cond in conditions:
            inverted_eq = SymmetricEq(-cond.lhs, -cond.rhs)
            if inverted_eq in conditions:
                conditions.discard(cond)
                return SingularityDetection._remove_duplicate_conditions(conditions)

        for cond1 in conditions:
            for cond2 in conditions:
                if cond1 == cond2:
                    continue

                if SingularityDetection._conditions_are_equivalent(cond1, cond2):
                    # keep the simplest condition; break ties deterministically by string representation
                    cond_to_discard = max([cond1, cond2], key=lambda cond: (sympy.count_ops(cond.rhs), str(cond.lhs), str(cond.rhs)))
                    conditions.discard(cond_to_discard)
                    return SingularityDetection._remove_duplicate_conditions(conditions)

        # nothing was removed
        return conditions

    @staticmethod
    def find_propagator_singularities(P: sympy.Matrix, A: sympy.Matrix) -> Set[SymmetricEq]:
        r"""Find singularities in the propagator matrix :math:`P` given the system matrix :math:`A`.

        Parameters
        ----------
        P : sympy.Matrix
            propagator matrix to check for singularities
        A : sympy.Matrix
            system matrix


        Returns
        -------

        conditions
            a set with equations, where the left-hand side of each equation is the variable that is to be substituted, and the right-hand side is the expression to put in its place
        """
        logging.getLogger(__name__).debug("Checking for singularities (divisions by zero) in the propagator matrix...")
        try:
            conditions = SingularityDetection._generate_singularity_conditions(P)
            conditions = SingularityDetection._filter_valid_conditions(conditions, A)  # filters out the invalid conditions (invalid means those for which A is not defined)
            conditions = SingularityDetection._remove_duplicate_conditions(conditions)
        except Exception as e:
            print(e)
            raise SingularityDetectionException()

        return conditions
