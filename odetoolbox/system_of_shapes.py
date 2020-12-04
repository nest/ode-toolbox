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

import functools
import json
import logging
import numpy as np
import sympy
import sympy.matrices

from .shapes import Shape


def _is_zero(x):
    r"""
    In the ideal case, we would like to use sympy.simplify() to do simplification of an expression before comparing it to zero. However, for expressions of moderate size (e.g. a few dozen terms involving exp() functions), it becomes unbearably slow. We therefore use this internal function, so that the simplification function can be easily switched over.

    Tests by expand_mul only, suitable for polynomials and rational functions.

    Ref.: https://github.com/sympy/sympy PR #13877 by @normalhuman et al. merged on Jan 27, 2018
    """
    return bool(sympy.expand_mul(x).is_zero)


class SystemOfShapes(object):
    r"""
    Represent a dynamical system in the canonical form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{c}`.
    """
    def __init__(self, x, A: sympy.Matrix, c, shapes):
        r"""
        Initialize a dynamical system in the canonical form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{c}`.

        :param A: Matrix containing linear part.
        :param c: Vector containing nonlinear part.
        """
        logging.debug("Initializing system of shapes with x = " + str(x) + ", A = " + str(A) + ", c = " + str(c))
        assert x.shape[0] == A.shape[0] == A.shape[1] == c.shape[0]
        self.x_ = x
        self.A_ = A
        self.c_ = c
        self.shapes_ = shapes


    def get_initial_value(self, sym):
        for shape in self.shapes_:
            if str(shape.symbol) == str(sym).replace("__d", "").replace("'", ""):
                return shape.get_initial_value(sym.replace("__d", "'"))
        assert False, "Unknown symbol: " + str(sym)


    def get_dependency_edges(self):
        E = []

        for i, sym1 in enumerate(self.x_):
            for j, sym2 in enumerate(self.x_):
                if not _is_zero(self.A_[j, i]):
                    E.append((sym2, sym1))
                else:
                    if not _is_zero(sympy.diff(self.c_[j], sym1)):
                        E.append((sym2, sym1))

        return E


    def get_lin_cc_symbols(self, E, differential_order_symbol="__d"):
        r"""
        Retrieve the variable symbols of those shapes that are linear and constant coefficient. In the case of a higher-order shape, will return all the variable symbols with ``"__d"`` suffixes up to the order of the shape.
        """
        node_is_lin = {}
        for shape in self.shapes_:
            if shape.is_lin_const_coeff(self.shapes_) and shape.is_homogeneous(self.shapes_):
                _node_is_lin = True
            else:
                _node_is_lin = False
            all_shape_symbols = shape.get_state_variables(derivative_symbol=differential_order_symbol)
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

        c_old = self.c_.copy()
        for _idx in idx:
            c_old[_idx] += self.A_[_idx, idx_compl].dot(self.x_[idx_compl, :])
            if len(str(c_old[_idx])) > Shape.EXPRESSION_SIMPLIFICATION_THRESHOLD:
                logging.warning("Skipping simplification of an expression that exceeds sympy simplification threshold")
            else:
                c_old[_idx] = sympy.simplify(c_old[_idx])

        c_sub = c_old[idx, :]

        shapes_sub = [shape for shape in self.shapes_ if shape.symbol in symbols]

        return SystemOfShapes(x_sub, A_sub, c_sub, shapes_sub)


    def generate_propagator_solver(self, output_timestep_symbol="__h"):
        r"""
        Generate the propagator matrix and symbolic expressions for propagator-based updates; return as JSON.
        """
        #
        #   generate the propagator matrix
        #

        P = sympy.simplify(sympy.exp(self.A_ * sympy.Symbol(output_timestep_symbol)))

        if sympy.I in sympy.preorder_traversal(P):
            raise Exception("The imaginary unit was found in the propagator matrix. This can happen if the dynamical system that was passed to ode-toolbox is unstable, i.e. one or more state variables will diverge to minus or positive infinity.")


        #
        #   generate symbols for each nonzero entry of the propagator matrix
        #

        P_sym = sympy.zeros(*P.shape)   # each entry in the propagator matrix is assigned its own symbol
        P_expr = {}     # the expression corresponding to each propagator symbol
        update_expr = {}    # keys are str(variable symbol), values are str(expressions) that evaluate to the new value of the corresponding key
        for row in range(P_sym.shape[0]):
            update_expr_terms = []
            for col in range(P_sym.shape[1]):
                if not _is_zero(P[row, col]):
                    sym_str = "__P__{}__{}".format(str(self.x_[row]), str(self.x_[col]))
                    P_sym[row, col] = sympy.parsing.sympy_parser.parse_expr(sym_str, global_dict=Shape._sympy_globals)
                    P_expr[sym_str] = P[row, col]
                    update_expr_terms.append(sym_str + " * " + str(self.x_[col]))
            update_expr[str(self.x_[row])] = " + ".join(update_expr_terms) + " + " + str(self.c_[row])
            update_expr[str(self.x_[row])] = sympy.simplify(sympy.parsing.sympy_parser.parse_expr(update_expr[str(self.x_[row])], global_dict=Shape._sympy_globals))

        all_state_symbols = [str(sym) for sym in self.x_]

        initial_values = {sym: str(self.get_initial_value(sym)) for sym in all_state_symbols}

        solver_dict = {"propagators": P_expr,
                       "update_expressions": update_expr,
                       "state_variables": all_state_symbols,
                       "initial_values": initial_values}

        return solver_dict


    def generate_numeric_solver(self):
        r"""
        Generate the symbolic expressions for numeric integration state updates; return as JSON.
        """
        update_expr = self.reconstitute_expr()
        all_state_symbols = [str(sym) for sym in self.x_]
        initial_values = {sym: str(self.get_initial_value(sym)) for sym in all_state_symbols}

        solver_dict = {"update_expressions": update_expr,
                       "state_variables": all_state_symbols,
                       "initial_values": initial_values}

        return solver_dict


    def reconstitute_expr(self):
        r"""
        Reconstitute a sympy expression from a system of shapes (which is internally encoded in the form Ax + c).
        """
        update_expr = {}

        for row, x in enumerate(self.x_):
            update_expr_terms = []
            for col, y in enumerate(self.x_):
                update_expr_terms.append(str(y) + " * (" + str(self.A_[row, col]) + ")")
            update_expr[str(x)] = " + ".join(update_expr_terms) + " + (" + str(self.c_[row]) + ")"
            update_expr[str(x)] = sympy.parsing.sympy_parser.parse_expr(update_expr[str(x)], global_dict=Shape._sympy_globals)
            if len(str(update_expr[str(x)])) > Shape.EXPRESSION_SIMPLIFICATION_THRESHOLD:
                logging.warning("Shape \"" + str(x) + "\" initialised with an expression that exceeds sympy simplification threshold")
            else:
                update_expr[str(x)] = sympy.simplify(update_expr[str(x)])

        return update_expr


    @classmethod
    def from_shapes(cls, shapes):
        r"""
        Construct the global system matrix :math:`\mathbf{A}` and nonlinear parts vector :math:`\mathbf{c}` on the basis of all shapes in ``shapes``, and such that

        .. math::

           \mathbf{x}' = \mathbf{Ax} + \mathbf{c}

        """
        if len(shapes) == 0:
            N = 0
        else:
            N = np.sum([shape.order for shape in shapes]).__index__()

        x = sympy.zeros(N, 1)
        A = sympy.zeros(N, N)
        c = sympy.zeros(N, 1)

        i = 0
        for shape in shapes:
            for j in range(shape.order):
                x[i] = shape.get_state_variables(derivative_symbol="__d")[j]
                i += 1

        i = 0
        for shape in shapes:
            highest_diff_sym_idx = [k for k, el in enumerate(x) if el == sympy.Symbol(str(shape.symbol) + "__d" * (shape.order - 1))][0]
            shape_expr = shape.reconstitute_expr()


            #
            #   grab the defining expression and separate into linear and nonlinear part
            #

            lin_factors, nonlin_term = Shape.split_lin_nonlin(shape_expr, x)
            A[highest_diff_sym_idx, :] = lin_factors[np.newaxis, :]
            c[highest_diff_sym_idx] = nonlin_term


            #
            #   for higher-order shapes: mark derivatives x_i' = x_(i+1) for i < shape.order
            #

            for order in range(shape.order - 1):
                A[i + order, i + order + 1] = 1.     # the n-th order derivative is at row n, starting at 0, until you reach the variable symbol without any "__d" suffixes

            i += shape.order

        return SystemOfShapes(x, A, c, shapes)
