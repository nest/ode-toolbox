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

import json

from sympy import diff, exp, Matrix, simplify, sqrt, Symbol, sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.matrices import zeros
import sympy
import sympy.matrices
import numpy
import numpy as np

from .shapes import Shape

class SystemOfShapes(object):
    """
    
    
    """

    def __init__(self, x, A, C, shapes):
        """
        Parameters
        ----------
        A : sympy.Matrix
            Jacobian of the system (square matrix).
        """
        assert x.shape[0] == A.shape[0] == A.shape[1] == C.shape[0]
        self.x_ = x
        self.A_ = A
        self.C_ = C
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
                if not sympy.simplify(self.A_[j, i]).is_zero:
                    E.append((sym2, sym1))
                    #E.append((str(sym2).replace("__d", "'"), str(sym1).replace("__d", "'")))
                else:
                    if not sympy.simplify(sympy.diff(self.C_[j], sym1)).is_zero:
                        E.append((sym2, sym1))
                        #E.append((str(sym2).replace("__d", "'"), str(sym1).replace("__d", "'")))

        return E


    def get_lin_cc_symbols(self, E):
        """retrieve the variable symbols of those shapes than are linear and constant coefficient. In the case of a higher-order shape, will return all the variable symbols with "__d" suffixes up to the order of the shape."""
        
        #
        # initial pass: is a node linear and constant coefficient by itself?
        #
        
        node_is_lin = {}
        for shape in self.shapes_:
            if shape.is_lin_const_coeff(self.shapes_):
                _node_is_lin = True
            else:
                _node_is_lin = False
            all_shape_symbols = [ sympy.Symbol(str(shape.symbol) + "__d" * i) for i in range(shape.order) ]
            for sym in all_shape_symbols:
                node_is_lin[sym] = _node_is_lin

        return node_is_lin


    def propagate_lin_cc_judgements(self, node_is_lin, E):
        """propagate: if a node depends on a node that is not linear and constant coefficient, it cannot be linear and constant coefficient"""
        
        queue = [ sym for sym, is_lin_cc in node_is_lin.items() if not is_lin_cc ]
        while len(queue) > 0:

            n = queue.pop(0)

            if not node_is_lin[n]:
                # mark dependent neighbours as also not lin_cc
                dependent_neighbours = [ n1 for (n1, n2) in E if n2 == n ]    # nodes that depend on n
                for n_neigh in dependent_neighbours:
                    if node_is_lin[n_neigh]:
                        #print("\t\tMarking dependent node " + str(n_neigh))
                        node_is_lin[n_neigh] = False
                        queue.append(n_neigh)

        return node_is_lin


    def get_jacobian_matrix(self):
        """Get the Jacobian matrix
        """
        N = len(self.x_)
        J = sympy.zeros(N, N)
        for i, sym in enumerate(self.x_):
            expr = self.C_[i]
            for v in self.A_[i, :]:
                expr += v
            for j, sym2 in enumerate(self.x_):
                J[i, j] = sympy.diff(expr, sym2)
        return J


    def get_sub_system(self, symbols):
        """Return a new instance which discards all symbols and equations except for those in `symbols`. This is probably only sensible when the elements in `symbols` do not dependend on any of the other symbols that will be thrown away.
        """

        idx = [ i for i, sym in enumerate(self.x_) if sym in symbols ]
        idx_compl = [ i for i, sym in enumerate(self.x_) if not sym in symbols ]
        
        x_sub = self.x_[idx, :]
        A_sub = self.A_[idx, :][:, idx]

        C_old = self.C_.copy()
        for _idx in idx:
            C_old[_idx] += self.A_[_idx, idx_compl].dot(self.x_[idx_compl, :])
            C_old[_idx] = sympy.simplify(C_old[_idx])
        
        C_sub = C_old[idx, :]
        
        shapes_sub = [shape for shape in self.shapes_ if shape.symbol in symbols]
        
        return SystemOfShapes(x_sub, A_sub, C_sub, shapes_sub)


    def generate_propagator_solver(self, output_timestep_symbol="__h"):
        """
        """

        #from IPython import embed;embed()

        #
        #   generate the propagator matrix
        #

        P = sympy.simplify(sympy.exp(self.A_ * sympy.Symbol(output_timestep_symbol)))
        

        #
        #   generate symbols for each nonzero entry of the propagator matrix
        #

        P_sym = sympy.zeros(*P.shape)   # each entry in the propagator matrix is assigned its own symbol
        P_expr = {}     # the expression corresponding to each propagator symbol
        update_expr = {}    # keys are str(variable symbol), values are str(expressions) that evaluate to the new value of the corresponding key
        for row in range(P_sym.shape[0]):
            update_expr_terms = []
            for col in range(P_sym.shape[1]):
                if sympy.simplify(P[row, col]) != sympy.sympify(0):
                    #sym_str = "__P_{}__{}_{}".format(self.x_[row], row, col)
                    sym_str = "__P__{}__{}".format(str(self.x_[row]), str(self.x_[col]))
                    P_sym[row, col] = sympy.parsing.sympy_parser.parse_expr(sym_str, global_dict=Shape._sympy_globals)
                    P_expr[sym_str] = P[row, col]
                    update_expr_terms.append(sym_str + " * " + str(self.x_[col]))
            update_expr[str(self.x_[row])] = " + ".join(update_expr_terms) + " + " + str(self.C_[row])
            update_expr[str(self.x_[row])] = sympy.simplify(sympy.parsing.sympy_parser.parse_expr(update_expr[str(self.x_[row])], global_dict=Shape._sympy_globals))
                    
        all_variable_symbols = [ str(sym) for sym in self.x_ ]

        initial_values = { sym : str(self.get_initial_value(sym)) for sym in all_variable_symbols }

        solver_dict = {"propagators" : P_expr,
                       "update_expressions" : update_expr,
                       "state_variables" : all_variable_symbols,
                       "initial_values" : initial_values}

        return solver_dict


    def generate_numeric_solver(self):
        """
        """
        
        update_expr = {}
        for row, x in enumerate(self.x_):
            update_expr_terms = []
            for col, y in enumerate(self.x_):
                update_expr_terms.append(str(y) + " * (" + str(self.A_[row, col]) + ")")
            update_expr[str(x)] = " + ".join(update_expr_terms) + " + (" + str(self.C_[row]) + ")"
            update_expr[str(x)] = sympy.simplify(sympy.parsing.sympy_parser.parse_expr(update_expr[str(x)], global_dict=Shape._sympy_globals))
        
        all_variable_symbols = [ str(sym) for sym in self.x_ ]
        initial_values = { sym : str(self.get_initial_value(sym)) for sym in all_variable_symbols }
        
        solver_dict = {"update_expressions" : update_expr,
                       "state_variables" : all_variable_symbols,
                       "initial_values" : initial_values}

        return solver_dict


    @classmethod
    def from_shapes(cls, shapes):
        """Construct the global system matrix including all shapes.
        
        Global dynamics
        
        .. math::
        
            x' = Ax + C

        where :math:`x` and :math:`C` are column vectors of length :math:`N` and :math:`A` is an :math:`N \times N` matrix.        
        """
        
        N = np.sum([shape.order for shape in shapes]).__index__()
        x = sympy.zeros(N, 1)
        A = sympy.zeros(N, N)
        C = sympy.zeros(N, 1)

        i = 0
        for shape in shapes:
            for j in range(shape.order):
                x[i] = shape.state_variables[j]
                i += 1

        i = 0
        for shape in shapes:
            #print("Shape: " + str(shape.symbol))
            highest_diff_sym_idx = [k for k, el in enumerate(x) if el == Symbol(str(shape.symbol) + "__d" * (shape.order - 1))][0]
            shape_expr = shape.diff_rhs_derivatives
            derivative_symbols = [ Symbol(str(shape.symbol) + "__d" * order) for order in range(shape.order) ]
            for derivative_factor, derivative_symbol in zip(shape.derivative_factors, derivative_symbols):
                shape_expr += derivative_factor * derivative_symbol
            #print("\t expr =  " + str(shape_expr))


            #
            #   grab the defining expression and separate into linear and nonlinear part
            #

            for j, sym1 in enumerate(x):
                diff_expr = sympy.simplify(sympy.diff(shape_expr, sym1))
                #print("\tdiff wrt " + str(sym1) + " = " + str(diff_expr))
                for sym2 in x:
                    #print("\t\tsym2 = " + str(sym2))
                    diff_wrt_sym2 = sympy.diff(diff_expr, sym2)
                    #print("\t\tdiff_wrt_sym2 = " + str(diff_wrt_sym2))
                    if not diff_wrt_sym2.is_zero:
                        # nonlinear term containing sym1
                        C[highest_diff_sym_idx] += sym1 * sym2 * diff_wrt_sym2
                        shape_expr -= sym1 * sym2 * diff_wrt_sym2                    
                        diff_expr -= sym2 * diff_wrt_sym2
                    shape_expr = sympy.simplify(shape_expr)
                    diff_expr = sympy.simplify(diff_expr)
                    A[highest_diff_sym_idx, j] = diff_expr
                    #print("\t\t---> new diff_expr = " + str(diff_expr))
                    #print("\t\t---> new shape_expr = " + str(shape_expr))

            # for higher-order shapes: mark subsequent derivatives x_i' = x_(i+1)
            for order in range(shape.order - 1):
                _idx = [k for k, el in enumerate(x) if el == Symbol(str(shape.symbol) + "__d" * (order + 1))][0]
                #print("\t\tThe symbol " + str(Symbol(str(shape.symbol) + "__d" * (order ))) + " is at position " + str(_idx) + " in vector " + str(x) + ", writing in row " + str(_idx))
                A[i + (shape.order - order - 1), _idx] = 1.     # the highest derivative is at row `i`, the next highest is below, and so on, until you reach the variable symbol without any "__d" suffixes

            i += shape.order

        return SystemOfShapes(x, A, C, shapes)


