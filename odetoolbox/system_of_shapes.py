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

class SystemOfShapes(object):
    """    """

    def __init__(self, shapes, output_timestep_symbol_name="__h"):
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
            shape_expr = shape.diff_rhs_derivatives
            derivative_symbols = [ Symbol(str(shape.symbol) + "__d" * order) for order in range(shape.order) ]
            for derivative_factor, derivative_symbol in zip(shape.derivative_factors, derivative_symbols):
                shape_expr = shape_expr + derivative_factor * derivative_symbol
            #print("\t expr =  " + str(shape_expr))

            highest_diff_sym_idx = [k for k, el in enumerate(x) if el == Symbol(str(shape.symbol) + "__d" * (shape.order - 1))][0]
            for j in range(N):
                A[highest_diff_sym_idx, j] = diff(shape_expr, x[j])
            
            # for higher-order shapes: mark subsequent derivatives x_i' = x_(i+1)
            for order in range(shape.order - 1):
                _idx = [k for k, el in enumerate(x) if el == Symbol(str(shape.symbol) + "__d" * (order + 1))][0]
                #print("\t\tThe symbol " + str(Symbol(str(shape.symbol) + "__d" * (order ))) + " is at position " + str(_idx) + " in vector " + str(x) + ", writing in row " + str(_idx))
                A[i + (shape.order - order - 1), _idx] = 1.     # the highest derivative is at row `i`, the next highest is below, and so on, until you reach the variable symbol without any "__d" suffixes

            i += shape.order
 
        i = 0
        for shape in shapes:
            print("Shape: " + str(shape.symbol))
            shape_expr = shape.diff_rhs_derivatives

            highest_diff_sym_idx = [k for k, el in enumerate(x) if el == Symbol(str(shape.symbol) + "__d" * (shape.order - 1))][0]
            for j in range(N):
                shape_expr = simplify(shape_expr - diff(shape_expr, x[j]) * x[j])

            C[highest_diff_sym_idx] = shape_expr

            i += shape.order
 
        #print("Matrices:")
        #print("x = " + str(x))
        #print("C = " + str(C))
        #print("A = " + str(A))

        self.x_ = x
        self.A_ = A
        self.C_ = C

        self.shapes_ = shapes
        

    def get_dependency_edges(self):

        E = []
        
        for i, sym1 in enumerate(self.x_):
            for j, sym2 in enumerate(self.x_):
                if not sympy.simplify(self.A_[j, i]) == sympy.parsing.sympy_parser.parse_expr("0"):
                    E.append((sym2, sym1))
                    #E.append((str(sym2).replace("__d", "'"), str(sym1).replace("__d", "'")))
                else:
                    if not sympy.simplify(sympy.diff(self.C_[j], sym1)) == sympy.parsing.sympy_parser.parse_expr("0"):
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

        #
        # propagate: if a node depends on a node that is not linear and constant coefficient, it cannot be linear and constant coefficient
        #

        queue = [ sym for sym, is_lin_cc in node_is_lin.items() if not is_lin_cc ]
        while len(queue) > 0:

            n = queue.pop(0)

            if not node_is_lin[n]:
                # mark dependent neighbours as also not lin_cc
                dependent_neighbours = [ n1 for (n1, n2) in E if n2 == n ]    # nodes that depend on n
                for n_neigh in dependent_neighbours:
                    if node_is_lin[n_neigh]:
                        print("\t\tMarking dependent node " + str(n_neigh))
                        node_is_lin[n_neigh] = False
                        queue.append(n_neigh)

        return node_is_lin
    
