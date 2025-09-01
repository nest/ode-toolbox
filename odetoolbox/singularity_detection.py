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
from typing import Dict, List, Set

import logging
import sympy
import sympy.parsing.sympy_parser

from odetoolbox.sympy_helpers import SymmetricEq, symbol_in_expression


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
    def _is_matrix_defined_under_substitution(A: sympy.Matrix, cond_set: Set[SymmetricEq]) -> bool:
        r"""
        Function to check if a matrix is defined (i.e. does not contain NaN or infinity) after we perform a given set of subsitutions.

        Parameters
        ----------
        A : sympy.Matrix
            input matrix
        cond_set : Set(SymmetricEq)
            a set with equations, where the left-hand side of each equation is the variable that is to be subsituted, and the right-hand side is the expression to put in its place
        """
        for val in sympy.flatten(A):
            expr_sub = val.copy()
            for eq in cond_set:
                expr_sub = expr_sub.subs(eq.lhs, eq.rhs)

            if symbol_in_expression([sympy.nan, sympy.zoo, sympy.oo], sympy.simplify(expr_sub)):
                return False

        return True

    @staticmethod
    def _filter_valid_conditions(conds, A: sympy.Matrix):
        filt_cond = set()
        for cond_set in conds:  # looping over condition sets
            if SingularityDetection._is_matrix_defined_under_substitution(A, cond_set):
                filt_cond.add(cond_set)

        return filt_cond

    @staticmethod
    def _generate_singularity_conditions(A: sympy.Matrix) -> List[Dict[sympy.core.expr.Expr, sympy.core.expr.Expr]]:
        r"""
        The function solve returns a list where each element is a dictionary. And each dictionary entry (condition: expression) corresponds to a condition at which that expression goes to zero. If the expression is quadratic, like let's say "x**2-1" then the function 'solve() returns two dictionaries in a list. Each dictionary corresponds to one solution. We are then collecting these lists in our own list called ``conditions`` and return it.
        """
        conditions = set()
        for expr in sympy.flatten(A):
            for subexpr in sympy.preorder_traversal(expr):  # traversing through the tree
                if isinstance(subexpr, sympy.Pow) and subexpr.args[1] < 0:  # find expressions of the form 1/x, which is encoded in sympy as x^-1
                    denom = subexpr.args[0]  # extracting the denominator
                    symbols = list(denom.free_symbols)
                    conds = SingularityDetection.find_singularity_conditions_in_expression_(denom, symbols)
                    conditions = conditions.union(conds)

        return conditions

    @staticmethod
    def find_singularity_conditions_in_expression_(expr, symbols) -> Set[SymmetricEq]:
        # find all conditions under which the denominator goes to zero. Each element of the returned list contains a particular combination of conditions for which A[row, row] goes to zero. For instance: ``solve([x - 3, y**2 - 1])`` returns ``[{x: 3, y: -1}, {x: 3, y: 1}]``
        conditions = sympy.solve(expr, symbols, dict=True, domain=sympy.S.Reals)

        # remove solutions that contain the imaginary number. ``domain=sympy.S.Reals`` does not seem to work perfectly as an argument to sympy.solve(), while sympy's ``reduce_inequalities()`` only supports univariate equations at the time of writing
        accepted_conditions = []
        for cond_set in conditions:
            i_in_expr = any([sympy.I in sympy.preorder_traversal(v) for v in cond_set.values()])
            if not i_in_expr:
                accepted_conditions.append(cond_set)

        conditions = accepted_conditions

        # convert dictionaries to sympy equations
        converted_conditions = set()
        for cond_set in conditions:
            cond_eqs_set = set([SymmetricEq(k, v) for k, v in cond_set.items()])    # convert to actual equations
            converted_conditions.add(frozenset(cond_eqs_set))

        conditions = converted_conditions

        return conditions

    @staticmethod
    def find_inhomogeneous_singularities(expr) -> Set[SymmetricEq]:
        r"""Find singularities in the inhomogeneous part of the update equations.

        Returns
        -------

        conditions
            a set with equations, where the left-hand side of each equation is the variable that is to be subsituted, and the right-hand side is the expression to put in its place
        """
        logging.debug("Checking for singularities (divisions by zero) in the inhomogeneous part of the update equations...")

        symbols = list(expr.free_symbols)
        conditions = set()
        if symbols:
            conditions = SingularityDetection.find_singularity_conditions_in_expression_(expr, symbols)
            if conditions:
                # if there is one or more condition under which the solution goes to infinity...

                logging.warning("Under certain conditions, one or more inhomogeneous term(s) in the system contain a division by zero.")
                logging.warning("List of all conditions that result in a division by zero:")
                for cond_set in conditions:
                    logging.warning("\t" + r" ∧ ".join([str(eq.lhs) + " = " + str(eq.rhs) for eq in cond_set]))

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
            a set with equations, where the left-hand side of each equation is the variable that is to be subsituted, and the right-hand side is the expression to put in its place
        """
        logging.debug("Checking for singularities (divisions by zero) in the propagator matrix...")
        try:
            conditions = SingularityDetection._generate_singularity_conditions(P)
            conditions = SingularityDetection._filter_valid_conditions(conditions, A)  # filters out the invalid conditions (invalid means those for which A is not defined)
        except Exception as e:
            print(e)
            raise SingularityDetectionException()

        return conditions
