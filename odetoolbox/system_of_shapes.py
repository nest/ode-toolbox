#
# system_of_shapes.py
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

import itertools
from typing import List, Optional

import logging
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import sympy
import sympy.matrices

from .config import Config
from .shapes import Shape
from .singularity_detection import SingularityDetection, SingularityDetectionException
from .sympy_helpers import _custom_simplify_expr, _is_zero


class GetBlockDiagonalException(Exception):
    """
    Thrown in case an error occurs while block diagonalising a matrix.
    """
    pass


def get_block_diagonal_blocks(A):
    assert A.shape[0] == A.shape[1], "matrix A should be square"

    A_connectivity_undirected = (A != 0) | (A.T != 0)    # make symmetric (undirected) connectivity graph from the system matrix

    graph_components = scipy.sparse.csgraph.connected_components(A_connectivity_undirected)[1]

    if not all(np.diff(graph_components) >= 0):
        # matrix is not ordered
        raise GetBlockDiagonalException()

    blocks = []
    for i in np.unique(graph_components):
        idx = np.where(graph_components == i)[0]

        if not all(np.diff(idx) > 0) or not (len(idx) == 1 or (len(np.unique(np.diff(idx))) == 1 and np.unique(np.diff(idx))[0] == 1)):
            raise GetBlockDiagonalException()

        idx_min = np.amin(idx)
        idx_max = np.amax(idx)
        block = A[idx_min:idx_max + 1, idx_min:idx_max + 1]
        blocks.append(block)

    return blocks


class PropagatorGenerationException(Exception):
    """
    Thrown in case an error occurs while generating propagators.
    """
    pass


class SystemOfShapes:
    r"""
    Represent a dynamical system in the canonical form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{b} + \mathbf{c}`.
    """
    def __init__(self, x: sympy.Matrix, A: sympy.Matrix, b: sympy.Matrix, c: sympy.Matrix, shapes: List[Shape]):
        r"""
        Initialize a dynamical system in the canonical form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{b} + \mathbf{c}`.

        :param x: Vector containing variable symbols.
        :param A: Matrix containing linear part.
        :param b: Vector containing inhomogeneous part (constant term).
        :param c: Vector containing nonlinear part.
        """
        logging.debug("Initializing system of shapes with x = " + str(x) + ", A = " + str(A) + ", b = " + str(b) + ", c = " + str(c))
        assert x.shape[0] == A.shape[0] == A.shape[1] == b.shape[0] == c.shape[0]
        self.x_ = x
        self.A_ = A
        self.b_ = b
        self.c_ = c
        self.shapes_ = shapes


    def get_shape_by_symbol(self, sym: str) -> Optional[Shape]:
        for shape in self.shapes_:
            if str(shape.symbol) == sym:
                return shape
        return None

    def get_initial_value(self, sym):
        for shape in self.shapes_:
            if str(shape.symbol) == str(sym).replace(Config().differential_order_symbol, "").replace("'", ""):
                return shape.get_initial_value(sym.replace(Config().differential_order_symbol, "'"))

        assert False, "Unknown symbol: " + str(sym)


    def get_dependency_edges(self):
        E = []

        for i, sym1 in enumerate(self.x_):
            for j, sym2 in enumerate(self.x_):
                if not _is_zero(self.A_[j, i]) or sym1 in self.c_[j].free_symbols:
                    E.append((sym2, sym1))

        return E


    def get_lin_cc_symbols(self, E, parameters=None):
        r"""
        Retrieve the variable symbols of those shapes that are linear and constant coefficient. In the case of a higher-order shape, will return all the variable symbols with ``"__d"`` suffixes up to the order of the shape.
        """
        # get all symbols for all shapes as a list
        symbols = list(self.x_)

        node_is_lin = {}
        for shape in self.shapes_:
            if shape.is_lin_const_coeff_in(symbols, parameters=parameters):
                _node_is_lin = True
            else:
                _node_is_lin = False
            all_shape_symbols = shape.get_state_variables(derivative_symbol=Config().differential_order_symbol)
            for sym in all_shape_symbols:
                node_is_lin[sym] = _node_is_lin

        return node_is_lin


    def propagate_lin_cc_judgements(self, node_is_lin, E):
        r"""
        Propagate: if a node depends on a node that is not linear and constant coefficient, it cannot be linear and constant coefficient.

        :param node_is_lin: Initial assumption about whether node is linear and constant coefficient.
        :param E: List of edges returned from dependency analysis.
        """
        queue = [sym for sym, is_lin_cc in node_is_lin.items() if not is_lin_cc]
        while len(queue) > 0:

            n = queue.pop(0)

            if not node_is_lin[n]:
                # mark dependent neighbours as also not lin_cc
                dependent_neighbours = [n1 for (n1, n2) in E if n2 == n]    # nodes that depend on n
                for n_neigh in dependent_neighbours:
                    if node_is_lin[n_neigh]:
                        node_is_lin[n_neigh] = False
                        queue.append(n_neigh)

        return node_is_lin


    def get_jacobian_matrix(self):
        r"""
        Get the Jacobian matrix as symbolic expressions. Entries in the matrix are sympy expressions.

        If the dynamics of variables :math:`x_1, \ldots, x_N` is defined as :math:`x_i' = f_i`, then row :math:`i` of the Jacobian matrix :math:`\mathbf{J}_i = \left[\begin{matrix}\frac{\partial f_i}{\partial x_0} & \cdots & \frac{\partial f_i}{\partial x_N}\end{matrix}\right]`.
        """
        N = len(self.x_)
        J = sympy.zeros(N, N)
        for i, sym in enumerate(self.x_):
            expr = self.c_[i]
            for v in self.A_[i, :]:
                expr += v
            for j, sym2 in enumerate(self.x_):
                J[i, j] = sympy.diff(expr, sym2)
        return J


    def get_sub_system(self, symbols):
        r"""
        Return a new :python:`SystemOfShapes` instance which discards all symbols and equations except for those in :python:`symbols`. This is probably only sensible when the elements in :python:`symbols` do not dependend on any of the other symbols that will be thrown away.
        """
        idx = [i for i, sym in enumerate(self.x_) if sym in symbols]
        idx_compl = [i for i, sym in enumerate(self.x_) if not sym in symbols]

        x_sub = self.x_[idx, :]
        A_sub = self.A_[idx, :][:, idx]
        b_sub = self.b_[idx, :]

        c_old = self.c_.copy()
        for _idx in idx:
            c_old[_idx] += self.A_[_idx, idx_compl].dot(self.x_[idx_compl, :])
            c_old[_idx] = _custom_simplify_expr(c_old[_idx])

        c_sub = c_old[idx, :]

        shapes_sub = [shape for shape in self.shapes_ if shape.symbol in symbols]

        return SystemOfShapes(x_sub, A_sub, b_sub, c_sub, shapes_sub)


    def _generate_propagator_matrix(self, A) -> sympy.Matrix:
        r"""Generate the propagator matrix by matrix exponentiation."""

        try:
            # optimized: be explicit about block diagonal elements; much faster!
            logging.debug("Computing propagator matrix (block-diagonal optimisation)...")
            blocks = get_block_diagonal_blocks(np.array(A))
            propagators = [sympy.simplify(sympy.exp(sympy.Matrix(block) * sympy.Symbol(Config().output_timestep_symbol))) for block in blocks]
            P = sympy.Matrix(scipy.linalg.block_diag(*propagators))
        except GetBlockDiagonalException:
            # naive: calculate propagators in one step
            logging.debug("Computing propagator matrix...")
            P = _custom_simplify_expr(sympy.exp(A * sympy.Symbol(Config().output_timestep_symbol)))

        # check the result
        if sympy.I in sympy.preorder_traversal(P):
            raise PropagatorGenerationException("The imaginary unit was found in the propagator matrix. This can happen if the dynamical system that was passed to ode-toolbox is unstable, i.e. one or more state variables will diverge to minus or positive infinity.")

        return P

    def _merge_conditions(self, solver_dict):
        r"""merge together conditions (a OR b OR c OR...) if the propagators and update_expressions are the same"""

        for condition, sub_solver_dict in solver_dict["conditions"].items():
            for condition2, sub_solver_dict2 in solver_dict["conditions"].items():
                if condition == condition2:
                    # don't check a condition against itself
                    continue

                if sub_solver_dict["propagators"] == sub_solver_dict2["propagators"] and sub_solver_dict["update_expressions"] == sub_solver_dict2["update_expressions"]:
                    # ``condition`` and ``condition2`` can be merged
                    solver_dict["conditions"]["(" + condition + ") || (" + condition2 + ")"] = sub_solver_dict
                    solver_dict["conditions"].pop(condition)
                    solver_dict["conditions"].pop(condition2)
                    return self._merge_conditions(solver_dict)

        return solver_dict

    def generate_propagator_solver(self, disable_singularity_detection: bool = False):
        r"""
        Generate the propagator matrix and symbolic expressions for propagator-based updates; return as JSON.
        """

        P = self._generate_propagator_matrix(self.A_)

        #
        #    singularity detection
        #

        if not disable_singularity_detection:
            # try:
            if 1:
                conditions = SingularityDetection.find_propagator_singularities(P, self.A_)
                conditions = conditions.union(SingularityDetection.find_inhomogeneous_singularities(self.A_, self.b_))

                if conditions:
                    # if there is one or more condition under which the solution goes to infinity...
                    logging.info("Under certain conditions, the default analytic solver contains singularities due to division by zero.")
                    logging.info("List of all conditions that result in a division by zero:")
                    for cond_set in conditions:
                        logging.info("\t" + r" ∧ ".join([str(eq.lhs) + " = " + str(eq.rhs) for eq in cond_set]))
                    logging.info("Alternate solvers will be generated for each of these conditions (and combinations thereof).")

                    # generate solver for the base case (with singularity conditions that are not met)
                    default_solver = self.generate_solver_dict_based_on_propagator_matrix_(P)

                    # change the returned solver dictionary to include conditions
                    solver_dict = {"solver": "analytical",
                                   "state_variables": default_solver["state_variables"],
                                   "initial_values": default_solver["initial_values"],
                                   "conditions": {"default": {"propagators": default_solver["propagators"],
                                                              "update_expressions": default_solver["update_expressions"]}}}

                    #
                    #    generate all combinations of conditions
                    #

                    num_conditions = len(conditions)
                    condition_permutations = list(itertools.product([False, True], repeat=num_conditions))
                    for condition_permutation in condition_permutations:
                        # each ``condition_permutation[i]`` is True/False corresponding to condition i

                        cond_set = set()    # cond_set is the set of conditions that have to hold
                        for i, cond in enumerate(condition_permutation):
                            if cond:
                                # all conditions in the set ``conditions[i]`` need to hold for this propagator
                                cond_set = cond_set.union(list(conditions)[i])
                            else:
                                # none of the conditions in the set ``conditions[i]`` are allowed to hold for this propagator
                                cond_set = cond_set.union(([sympy.Ne(eq.lhs, eq.rhs) for eq in list(conditions)[i]]))

                        condition_str: str = " && ".join(["(" + str(eq.lhs) + (" == " if isinstance(eq, sympy.Eq) else "!=") + str(eq.rhs) + ")" for eq in cond_set])

                        logging.debug("Generating solver for condition: " + str(condition_str))

                        if not any([isinstance(eq, sympy.Eq) for eq in cond_set]):
                            # this is the default condition, only containing inequalities
                            continue

                        conditional_A = self.A_.copy()
                        conditional_b = self.b_.copy()
                        conditional_c = self.c_.copy()

                        for eq in cond_set:
                            if isinstance(eq, sympy.Eq):
                                # replace equalities (not inequalities)
                                conditional_A = conditional_A.subs(eq.lhs, eq.rhs)
                                conditional_b = conditional_b.subs(eq.lhs, eq.rhs)
                                conditional_c = conditional_c.subs(eq.lhs, eq.rhs)

                        conditional_dynamics = SystemOfShapes(self.x_, conditional_A, conditional_b, conditional_c, self.shapes_)
                        solver_dict_conditional = conditional_dynamics.generate_propagator_solver(disable_singularity_detection=True)
                        solver_dict["conditions"][condition_str] = {"propagators": solver_dict_conditional["propagators"],
                                                                    "update_expressions": solver_dict_conditional["update_expressions"]}

                    solver_dict = self._merge_conditions(solver_dict)

                    return solver_dict

            # except SingularityDetectionException:
                # logging.warning("Could not check the propagator matrix for singularities.")

        return self.generate_solver_dict_based_on_propagator_matrix_(P)

    def generate_solver_dict_based_on_propagator_matrix_(self, P: sympy.Matrix):

        #
        #   generate symbols for each nonzero entry of the propagator matrix
        #

        P_expr = {}     # the expression corresponding to each propagator symbol
        update_expr = {}    # keys are str(variable symbol), values are str(expressions) that evaluate to the new value of the corresponding key
        for row in range(P.shape[0]):
            # assemble update expression for symbol ``self.x_[row]``
            if not _is_zero(self.c_[row]):
                raise PropagatorGenerationException("For symbol " + str(self.x_[row]) + ": nonlinear part should be zero for propagators")

            if not _is_zero(self.b_[row]) and self.shape_order_from_system_matrix(row) > 1:
                raise PropagatorGenerationException("For symbol " + str(self.x_[row]) + ": higher-order inhomogeneous ODEs are not supported")

            update_expr_terms = []
            for col in range(P.shape[1]):
                if not _is_zero(P[row, col]):
                    sym_str = Config().propagators_prefix + "__{}__{}".format(str(self.x_[row]), str(self.x_[col]))
                    P_expr[sym_str] = P[row, col]
                    if row != col and not _is_zero(self.b_[col]):
                        # the ODE for x_[row] depends on the inhomogeneous ODE of x_[col]. We can't solve this analytically in the general case (even though some specific cases might admit a solution)
                        raise PropagatorGenerationException("the ODE for " + str(self.x_[row]) + " depends on the inhomogeneous ODE of " + str(self.x_[col]) + ". We can't solve this analytically in the general case (even though some specific cases might admit a solution)")

                    update_expr_terms.append(sym_str + " * " + str(self.x_[col]))

            if not _is_zero(self.b_[row]):
                # this is an inhomogeneous ODE
                if _is_zero(self.A_[row, row]):
                    # of the form x' = const
                    update_expr_terms.append(Config().output_timestep_symbol + " * " + str(self.b_[row]))
                else:

                    #
                    #    generate update expressions
                    #

                    particular_solution = -self.b_[row] / self.A_[row, row]
                    sym_str = Config().propagators_prefix + "__{}__{}".format(str(self.x_[row]), str(self.x_[row]))
                    update_expr_terms.append("-" + sym_str + " * " + str(self.x_[row]))    # remove the term (add its inverse) that would have corresponded to a homogeneous solution and that was added in the ``for col...`` loop above
                    update_expr_terms.append(sym_str + " * (" + str(self.x_[row]) + " - (" + str(particular_solution) + "))" + " + (" + str(particular_solution) + ")")

            update_expr[str(self.x_[row])] = " + ".join(update_expr_terms)
            update_expr[str(self.x_[row])] = sympy.parsing.sympy_parser.parse_expr(update_expr[str(self.x_[row])], global_dict=Shape._sympy_globals)
            if not _is_zero(self.b_[row]):
                # only simplify in case an inhomogeneous term is present
                update_expr[str(self.x_[row])] = _custom_simplify_expr(update_expr[str(self.x_[row])])
            logging.debug("update_expr[" + str(self.x_[row]) + "] = " + str(update_expr[str(self.x_[row])]))

        all_state_symbols = [str(sym) for sym in self.x_]
        initial_values = {sym: str(self.get_initial_value(sym)) for sym in all_state_symbols}
        solver_dict = {"solver": "analytical",
                       "propagators": P_expr,
                       "update_expressions": update_expr,
                       "state_variables": all_state_symbols,
                       "initial_values": initial_values}

        return solver_dict


    def generate_numeric_solver(self, state_variables=None):
        r"""
        Generate the symbolic expressions for numeric integration state updates; return as JSON.
        """
        update_expr = self.reconstitute_expr(state_variables=state_variables)
        all_state_symbols = [str(sym) for sym in self.x_]
        initial_values = {sym: str(self.get_initial_value(sym)) for sym in all_state_symbols}

        solver_dict = {"solver": "numeric",   # will be appended to if stiffness testing is used
                       "update_expressions": update_expr,
                       "state_variables": all_state_symbols,
                       "initial_values": initial_values}

        return solver_dict


    def reconstitute_expr(self, state_variables=None):
        r"""
        Reconstitute a sympy expression from a system of shapes (which is internally encoded in the form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{b} + \mathbf{c}`).

        Before returning, the expression is simplified using a custom series of steps, passed via the ``simplify_expression`` argument (see the ODE-toolbox documentation for more details).
        """
        if state_variables is None:
            state_variables = []

        update_expr = {}

        for row, x in enumerate(self.x_):
            update_expr_terms = []
            for col, y in enumerate(self.x_):
                if str(self.A_[row, col]) in ["1", "1.", "1.0"]:
                    update_expr_terms.append(str(y))
                else:
                    update_expr_terms.append(str(y) + " * (" + str(self.A_[row, col]) + ")")
            update_expr[str(x)] = " + ".join(update_expr_terms) + " + (" + str(self.b_[row]) + ") + (" + str(self.c_[row]) + ")"
            update_expr[str(x)] = sympy.parsing.sympy_parser.parse_expr(update_expr[str(x)], global_dict=Shape._sympy_globals)

        # custom expression simplification
        for name, expr in update_expr.items():
            update_expr[name] = _custom_simplify_expr(expr)
            collect_syms = [sym for sym in update_expr[name].free_symbols if not (sym in state_variables or str(sym) in state_variables)]
            update_expr[name] = sympy.collect(update_expr[name], collect_syms)

        return update_expr


    def shape_order_from_system_matrix(self, idx: int) -> int:
        r"""Determine shape differential order from system matrix of symbol ``self.x_[idx]``"""
        N = self.A_.shape[0]
        A = np.zeros((N, N), dtype=int)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] = not _is_zero(self.A_[i, j])

        scc = scipy.sparse.csgraph.connected_components(A, connection="strong")[1]
        shape_order = sum(scc == scc[idx])
        return shape_order


    def get_connected_symbols(self, idx: int) -> List[sympy.Symbol]:
        r"""Extract all symbols belonging to a shape with symbol ``self.x_[idx]`` from the system matrix.

        For example, if symbol ``i`` is ``x``, and symbol ``j`` is ``y``, and the system is:

        .. math::

           \frac{dx}{dt} &= y\\
           \frac{dy}{dt} &= y' = -\frac{1}{\tau^2} x - \frac{2}{\tau} y

        Then ``get_connected_symbols()`` for symbol ``x`` would return ``[x, y]``, and ``get_connected_symbols()`` for ``y`` would return the same.
        """
        N = self.A_.shape[0]
        A = np.zeros((N, N), dtype=int)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] = not _is_zero(self.A_[i, j])

        scc = scipy.sparse.csgraph.connected_components(A, connection="strong")[1]
        idx = np.where(scc == scc[idx])[0]
        return [self.x_[i] for i in idx]


    @classmethod
    def from_shapes(cls, shapes: List[Shape], parameters=None):
        r"""
        Construct the global system matrix :math:`\mathbf{A}` and inhomogeneous part (constant term) :math:`\mathbf{b}` and nonlinear part :math:`\mathbf{c}` on the basis of all shapes in ``shapes``, and such that

        .. math::

           \mathbf{x}' = \mathbf{Ax} + \mathbf{b} + \mathbf{c}

        """
        if len(shapes) == 0:
            N = 0
        else:
            N = np.sum([shape.order for shape in shapes]).__index__()

        x = sympy.zeros(N, 1)
        A = sympy.zeros(N, N)
        b = sympy.zeros(N, 1)
        c = sympy.zeros(N, 1)

        i = 0
        for shape in shapes:
            for j in range(shape.order):
                x[i] = shape.get_state_variables(derivative_symbol=Config().differential_order_symbol)[j]
                i += 1

        i = 0
        for shape in shapes:
            highest_diff_sym_idx = [k for k, el in enumerate(x) if el == sympy.Symbol(str(shape.symbol) + Config().differential_order_symbol * (shape.order - 1))][0]
            shape_expr = shape.reconstitute_expr()

            #
            #   grab the defining expression and separate into linear and nonlinear part
            #

            lin_factors, inhom_term, nonlin_term = Shape.split_lin_inhom_nonlin(shape_expr, x, parameters=parameters)
            A[highest_diff_sym_idx, :] = lin_factors.T
            b[highest_diff_sym_idx] = inhom_term
            c[highest_diff_sym_idx] = nonlin_term


            #
            #   for higher-order shapes: mark derivatives x_i' = x_(i+1) for i < shape.order
            #

            for order in range(shape.order - 1):
                A[i + order, i + order + 1] = 1.     # the n-th order derivative is at row n, starting at 0, until you reach the variable symbol without any "__d" suffixes

            i += shape.order

        return SystemOfShapes(x, A, b, c, shapes)
