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
from typing import Mapping

import sympy
import sympy.parsing.sympy_parser


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
    def _is_matrix_defined_under_substitution(A: sympy.Matrix, cond: Mapping) -> bool:
        r"""
        Function to check if a matrix is defined (i.e. does not contain NaN or infinity) after we perform a given set of subsitutions.

        Parameters
        ----------
        A : sympy.Matrix
            input matrix
        cond : Mapping
            mapping from expression that is to be subsituted, to expression to put in its place
        """
        for val in sympy.flatten(A):
            for expr, subs_expr in cond.items():
                if sympy.simplify(val.subs(expr, subs_expr)) in [sympy.nan, sympy.zoo, sympy.oo]:
                    return False

        return True

    @staticmethod
    def _flatten_conditions(cond):
        r"""
        Return a list with conditions in the form of dictionaries
        """
        lst = []
        for i in range(len(cond)):
            if cond[i] not in lst:
                lst.append(cond[i])

        return lst

    @staticmethod
    def _filter_valid_conditions(cond, A: sympy.Matrix):
        filt_cond = []
        for i in range(len(cond)):  # looping over conditions
            if SingularityDetection._is_matrix_defined_under_substitution(A, cond[i]):
                filt_cond.append(cond[i])

        return filt_cond

    @staticmethod
    def _generate_singularity_conditions(A: sympy.Matrix):
        r"""
        The function solve returns a list where each element is a dictionary. And each dictionary entry (condition: expression) corresponds to a condition at which that expression goes to zero.
        If the expression is quadratic, like let's say "x**2-1" then the function 'solve() returns two dictionaries in a list. each dictionary corresponds to one solution.
        We are then collecting these lists in our own list called 'condition'.
        """
        conditions = []
        for expr in sympy.flatten(A):
            for subexpr in sympy.preorder_traversal(expr):  # traversing through the tree
                if isinstance(subexpr, sympy.Pow) and subexpr.args[1] < 0:  # find expressions of the form 1/x, which is encoded in sympy as x^-1
                    denom = subexpr.args[0]  # extracting the denominator
                    cond = sympy.solve(denom, denom.free_symbols, dict=True)  # ``cond`` here is a list of all those conditions at which the denominator goes to zero
                    if cond not in conditions:
                        conditions.extend(cond)

        return conditions

    @staticmethod
    def find_singularities(P: sympy.Matrix, A: sympy.Matrix):
        r"""Find singularities in the propagator matrix :math:`P` given the system matrix :math:`A`.

        Parameters
        ----------
        P : sympy.Matrix
            propagator matrix to check for singularities
        A : sympy.Matrix
            system matrix
        """
        try:
            conditions = SingularityDetection._generate_singularity_conditions(P)
            conditions = SingularityDetection._flatten_conditions(conditions)  # makes a list of conditions with each condition in the form of a dict
            conditions = SingularityDetection._filter_valid_conditions(conditions, A)  # filters out the invalid conditions (invalid means those for which A is not defined)
        except Exception as e:
            raise SingularityDetectionException()

        return conditions
