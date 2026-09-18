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
from typing import List, Optional, Set, Union

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
from .sympy_helpers import SymmetricEq, _custom_simplify_expr, _is_zero, expMt, _sympy_parse_real
from sympy.matrices.exceptions import NonInvertibleMatrixError


class GetBlockDiagonalException(Exception):
    """
    Thrown in case an error occurs while block diagonalising a matrix.
    """
    pass


def get_block_diagonal_blocks(A):

    # maps set of variables to the derivatives
    assert A.shape[0] == A.shape[1], "matrix A should be square"

    # make symmetric (undirected) connectivity graph from the system matrix
    A_connectivity_undirected = (A != 0) | (A.T != 0)

    # creating symmetric boolean matrix where an edge (1) exists if two
    # varibles influence eachother (undirected graph)
    graph_components = scipy.sparse.csgraph.connected_components(
        A_connectivity_undirected)[1]

    # reordering the diagonal blocks
    # if not all(np.diff(graph_components) >= 0): # if the blocks aren't contigious
    #     # Find the sorting map that groups matching component IDs together
    #     permutation = np.argsort(graph_components, kind="stable")

    #     # Re-arrange the rows and columns of the system matrix A in memory
    # A = A[np.ix_(permutation, permutation)] # use np.ix_ here because A is a
    # SymPy object matrix

    #     # update the component array so it matches our newly sorted matrix layout
    #     graph_components = graph_components[permutation]

    # checking for ordering and testing if blocks are contigious
    if not all(np.diff(graph_components) >= 0):
        raise GetBlockDiagonalException()

    blocks = []
    for i in np.unique(
            graph_components):  # code block for slicing out independent blocks
        idx = np.where(graph_components == i)[0]

        if not all(np.diff(idx) > 0) or not (len(idx) == 1 or (
                len(np.unique(np.diff(idx))) == 1 and np.unique(np.diff(idx))[0] == 1)):
            # checks for proximity of the blocks again for contigious input
            raise GetBlockDiagonalException()

        # for each isolated group of variables
        idx_min = np.amin(idx)
        idx_max = np.amax(idx)
        # assert assigned previously to ensure indices form a contiguous
        # sequence of ints
        block = A[idx_min:idx_max + 1, idx_min:idx_max + 1]
        blocks.append(block)

    return blocks  # diagonalized blocks returned

    # not passing permutations bacK?


class PropagatorGenerationException(Exception):
    """
    Thrown in case an error occurs while generating propagators.
    """
    pass


class SystemOfShapes:
    r"""
    Represent a dynamical system in the canonical form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{b} + \mathbf{c}`.
    """

    def __init__(
        self,
        x: sympy.Matrix,
        A: sympy.Matrix,
        b: sympy.Matrix,
        c: sympy.Matrix,
            shapes: List[Shape]):
        r"""
        Initialize a dynamical system in the canonical form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{b} + \mathbf{c}`.

        :param x: Vector containing variable symbols.
        :param A: Matrix containing linear part.
        :param b: Vector containing inhomogeneous part (constant term).
        :param c: Vector containing nonlinear part.
        """
        assert x.shape[0] == A.shape[0] == A.shape[1] == b.shape[0] == c.shape[0]
        self.x_ = x
        self.A_ = A
        self.b_ = b
        self.c_ = c
        self.shapes_ = shapes

    # utility look up for shape by symbol
    def get_shape_by_symbol(
            self, sym: Union[str, sympy.Symbol]) -> Optional[Shape]:
        for shape in self.shapes_:
            if str(shape.symbol) == str(sym):
                return shape

        return None

    # extracts default value of a variable. passing _P__V_m it searches for
    # V_m default value.
    def get_initial_value(self, sym: Union[str, sympy.Symbol]):
        for shape in self.shapes_:
            if str(
                shape.symbol) == str(sym).replace(
                Config().differential_order_symbol,
                "").replace(
                "'",
                    ""):
                return shape.get_initial_value(str(sym).replace(
                    Config().differential_order_symbol, "'"))

        assert False, "Unknown symbol: " + str(sym)

    # builds a directed graph of dependencies between your ODE varaibles.
    # scanning eqs to see which variables influence eachother
    def get_dependency_edges(self):
        E = []
        for i, sym1 in enumerate(self.x_):
            for j, sym2 in enumerate(self.x_):
                if not _is_zero(self.A_[j, i]) or sym1 in self.c_[
                        j].free_symbols:
                    E.append((sym2, sym1))

        return E

    def get_lin_cc_symbols(self, E, parameters=None):
        r"""
        Retrieve the variable symbols of those shapes that are linear and constant coefficient.
        In the case of a higher-order shape, will return all the variable symbols with ``"__d"`` suffixes up to the order of the shape.
        """
        # get all symbols for all shapes as a list
        symbols = list(self.x_)

        node_is_lin = {}
        for shape in self.shapes_:
            if shape.is_lin_const_coeff_in(symbols, parameters=parameters):
                _node_is_lin = True
            else:
                _node_is_lin = False
            all_shape_symbols = shape.get_state_variables(
                derivative_symbol=Config().differential_order_symbol)
            for sym in all_shape_symbols:
                node_is_lin[sym] = _node_is_lin

        return node_is_lin

    def propagate_lin_cc_judgements(self, node_is_lin, E):
        r"""
        Propagate: if a node depends on a node that is not linear and constant coefficient, it cannot be linear and constant coefficient.

        :param node_is_lin: Initial assumption about whether node is linear and constant coefficient.
        :param E: List of edges returned from dependency analysis.
        """
        queue = [sym for sym, is_lin_cc in node_is_lin.items()
                 if not is_lin_cc]
        while len(queue) > 0:

            n = queue.pop(0)

            if not node_is_lin[n]:
                # mark dependent neighbours as also not lin_cc
                # nodes that depend on n
                dependent_neighbours = [n1 for (n1, n2) in E if n2 == n]
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
        Return a new :python:`SystemOfShapes` instance which discards all symbols and equations except for those in :python:`symbols`.
            This is probably only sensible when the elements in :python:`symbols` do not dependend on any of the other symbols that will be thrown away.
        """
        idx = [i for i, sym in enumerate(self.x_) if sym in symbols]
        idx_compl = [i for i, sym in enumerate(self.x_) if sym not in symbols]

        x_sub = self.x_[idx, :]
        A_sub = self.A_[idx, :][:, idx]
        b_sub = self.b_[idx, :]

        c_old = self.c_.copy()
        for _idx in idx:
            c_old[_idx] += self.A_[_idx, idx_compl].dot(self.x_[idx_compl, :])
            c_old[_idx] = _custom_simplify_expr(c_old[_idx])

        c_sub = c_old[idx, :]

        shapes_sub = [
            shape for shape in self.shapes_ if shape.symbol in symbols]

        return SystemOfShapes(x_sub, A_sub, b_sub, c_sub, shapes_sub)

    def _generate_propagator_matrix(
            self, A, use_alternative_expM: bool = False) -> sympy.Matrix:
        r"""Generate the propagator matrix by matrix exponentiation."""

        if use_alternative_expM:  # computes matrix exponential
            expM = expMt
        else:
            expM = sympy.exp

        try:
            # optimized: compute propagators separately for each block diagonal
            # element of ``A``
            logging.getLogger(__name__).debug(
                "Computing propagator matrix (block-diagonal optimisation)...")

            # get diagnoal blocks per matrix diagonal element of ``A`` &&
            # perform reordering of components if needed to avoid
            # non-contigious output
            blocks = get_block_diagonal_blocks(np.array(A))

            # for block of matrix A calculate the propagator
            propagators = [
                _custom_simplify_expr(
                    expM(
                        sympy.Matrix(block) * sympy.Symbol(
                            Config().output_timestep_symbol,
                            real=True))) for block in blocks]

            P = sympy.Matrix(scipy.linalg.block_diag(*propagators))

        except GetBlockDiagonalException:
            # naive: calculate propagators in one step -- can be quite slow if
            # ``A`` is a large matrix
            logging.getLogger(__name__).debug("Computing propagator matrix...")
            P = _custom_simplify_expr(
                expM(
                    A * sympy.Symbol(
                        Config().output_timestep_symbol,
                        real=True)))

        # check the result
        if sympy.I in sympy.preorder_traversal(P):
            raise PropagatorGenerationException(
                "The imaginary unit was found in the propagator matrix. This can happen if the dynamical system that was passed to ode-toolbox is unstable, i.e. one or more state variables will diverge to minus or positive infinity.")

        return P

    def _merge_conditions(self, solver_dict):
        r"""merge together conditions (a OR b OR c OR...) if the propagators and update_expressions are the same"""

        for condition, sub_solver_dict in solver_dict["conditions"].items(
        ):  # combining matrix maths and updating code blocks, called recursively
            for condition2, sub_solver_dict2 in solver_dict["conditions"].items(
            ):
                if condition == condition2:
                    # don't check a condition against itself
                    continue

                if sub_solver_dict["propagators"] == sub_solver_dict2["propagators"] and sub_solver_dict[
                        "update_expressions"] == sub_solver_dict2["update_expressions"]:
                    # ``condition`` and ``condition2`` can be merged
                    solver_dict["conditions"]["(" + condition + ") || (" + condition2 + ")"] = sub_solver_dict
                    solver_dict["conditions"].pop(condition)
                    solver_dict["conditions"].pop(condition2)
                    return self._merge_conditions(solver_dict)

        return solver_dict

    def generate_propagator_solver(
            self,
            disable_singularity_detection: bool = False,
            disable_singularity_mitigation: bool = False,
            use_alternative_expM: bool = False):
        r"""
        Generate the propagator matrix and symbolic expressions for propagator-based updates; return as JSON.
        """

        P = self._generate_propagator_matrix(
            self.A_, use_alternative_expM=use_alternative_expM)

        #
        #    singularity detection
        #

        if not disable_singularity_detection:  # checks for singularities
            try:
                conditions = SingularityDetection.find_propagator_singularities(
                    P, self.A_)  # find singularities in the propagator matrix
                # find parameters sets to uncover where matrix calculation
                # collapses
                conditions = conditions.union(
                    SingularityDetection.find_inhomogeneous_singularities(
                        self.A_, self.b_))
                conditions = SingularityDetection._remove_duplicate_conditions(
                    conditions)

                if conditions and not disable_singularity_mitigation:

                    # generate default solver for the equation assuming that
                    # there are no singularity conditions
                    default_solver = self.generate_solver_dict_based_on_propagator_matrix_(
                        P)

                    # change the returned solver dictionary to include
                    # conditions
                    solver_dict = {
                        "solver": "analytical",
                        "state_variables": default_solver["state_variables"],
                        "initial_values": default_solver["initial_values"],
                        "conditions": {
                            "default": {
                                "propagators": default_solver["propagators"],
                                "update_expressions": default_solver["update_expressions"]}}}

                    #
                    #    generate all combinations of conditions
                    #

                    # ERROR propagation explosion occuring here for every
                    # permutation in the ode toolbox for singularities and
                    # applies equalties and creates a new system of shapes

                    # number of conditions of singularities we need to be aware
                    # of
                    num_conditions = len(conditions)
                    # maps out every possible combination of these conditions
                    # with a true/false
                    condition_permutations = list(itertools.product(
                        [False, True], repeat=num_conditions))

                    logging.getLogger(__name__).info(
                        "Alternate solvers will be generated for each of these conditions (and combinations thereof), which amounts to " + str(len(condition_permutations)) + " solvers that will be generated.")

                    # which of the current purmutations currently hold true
                    for condition_permutation in condition_permutations:
                        # each ``condition_permutation[i]`` is True/False
                        # corresponding to condition i

                        cond_set = set()    # cond_set is the set of conditions that have to hold
                        for i, cond_holds in enumerate(condition_permutation):
                            cond = list(conditions)[i]
                            if cond_holds:
                                # ``cond`` needs to hold for this propagator, since this currently condition will create an inequality
                                cond_set.add(cond)
                            else:
                                # ``cond`` needs to **not** hold for this propagator
                                cond_set.add(sympy.Ne(cond.lhs, cond.rhs))

                        condition_str: str = " && ".join(["(" + str(eq.lhs) + (" == " if isinstance(
                            eq, SymmetricEq) else "!=") + str(eq.rhs) + ")" for eq in cond_set])

                        if not any([isinstance(eq, SymmetricEq)
                                   for eq in cond_set]):
                            # this is the default condition, only containing
                            # inequalities
                            continue

                        logging.getLogger(__name__).debug(
                            "Generating solver for condition: " + str(condition_str))

                        conditional_A = self.A_.copy()
                        conditional_b = self.b_.copy()
                        conditional_c = self.c_.copy()

                        # ERROR creating impossible branches that the
                        # dependencies don't actually depend on leading to the
                        # explosion
                        for eq in cond_set:
                            if isinstance(eq, SymmetricEq):
                                # replace equalities (not inequalities)
                                conditional_A = conditional_A.subs(
                                    eq.lhs, eq.rhs)
                                conditional_b = conditional_b.subs(
                                    eq.lhs, eq.rhs)
                                conditional_c = conditional_c.subs(
                                    eq.lhs, eq.rhs)

                        conditional_dynamics = SystemOfShapes(
                            self.x_, conditional_A, conditional_b, conditional_c, self.shapes_)
                        solver_dict_conditional = conditional_dynamics.generate_propagator_solver(
                            disable_singularity_detection=True,
                            disable_singularity_mitigation=True,
                            use_alternative_expM=use_alternative_expM)
                        solver_dict["conditions"][condition_str] = {
                            "propagators": solver_dict_conditional["propagators"],
                            "update_expressions": solver_dict_conditional["update_expressions"]}

                    # cleans up after the expensive combinations have been created
                    # finds branches with identical propagators and
                    # update_conditions explains enormous condition keys in
                    # json
                    # simplifies logic scans through generated dict structure
                    # before exported to JSON template
                    solver_dict = self._merge_conditions(solver_dict)

                    return solver_dict

            except SingularityDetectionException:
                logging.getLogger(__name__).warning(
                    "Could not check the propagator matrix for singularities.")

        return self.generate_solver_dict_based_on_propagator_matrix_(P)






    def generate_solver_dict_based_on_propagator_matrix_(self, P: sympy.Matrix):
        """
        generate the analytical solver from a linear ode propagator matrix. Computes block-wise for coupled systems using 
        inverse of block or via constant-drift equations for isolated quations 

        """
        
        #
        # generate symbols for each nonzero entry of the propagator matrix
        #

        P_expr = {}  # the expression corresponding to each propagator symbol
        # keys are str(variable symbol), values are str(expressions) that
        # evaluate to the new value of the corresponding key
        update_expr = {}

        particular_solutions, constant_drift_rows = ( self._get_particular_solutions_for_coupled_components_())

        

        for row in range(self.A_.shape[0]):
            for col in range(self.A_.shape[1]):

                # build a connectivity matrix from A_ marking which state
                # variables are linked by a nonzero coefficient
                if (not _is_zero(self.A_[row, col]) or not _is_zero(self.A_[col, row])):
                    connectivity[row, col] = 1

        # find connected components, grouping state variables into clusters
        # that are coupled
        _, component_labels = scipy.sparse.csgraph.connected_components(
            scipy.sparse.csr_matrix(connectivity),
            directed=False)

        # Compute a particular solution for individual coupled inhomogeneous
        # blocks
        particular_solutions = {}
        
        # initialise the constant drift dictionary
        constant_drift_rows = set()

        for component in set(
                component_labels):  # for each coupled inhomogenous blocks

            indices = [i for i, label in enumerate(
                component_labels) if label == component]

            A_block = self.A_.extract(indices, indices)
            b_block = self.b_.extract(indices, [0])

            # Homogeneous blocks do not require a particular solution.
            if all(_is_zero(b_block[i, 0]) for i in range(len(indices))):
                continue

            # Isolated equation of the form, checks for steady state
            if (len(indices) == 1 and _is_zero(A_block[0, 0])):
                
                # if matrix is non-invertible, append to constant drift dict
                constant_drift_rows.add(indices[0])
                continue

            try:  # extracts the corresponding sub-matrix/sub-vector and solves the whole block via matrix inversion
                x_particular = -(A_block.inv() * b_block)

            # if a blocks matrix is not invertible, eq may not have the inverse
            # of matrix A
            except NonInvertibleMatrixError as exc:
                raise PropagatorGenerationException(
                    "Could not compute a particular solution for the coupled inhomogeneous system containing: " + ", ".join(str(self.x_[i]) for i in indices)) from exc

            for local_idx, global_idx in enumerate(indices):
                particular_solutions, constant_drift_rows = (
                    self._get_particular_solutions_for_coupled_components_()
                )

        # generate symbolic propagators and construct the state update 
        for row in range(P.shape[0]):
            if not _is_zero(self.c_[row]):
                raise PropagatorGenerationException("For symbol " + str(self.x_[row]) + ": nonlinear part should be zero for propagators")

            # guards against higher-order inhomogenous ODE's having a non zero
            # constant
            if (not _is_zero(
                    self.b_[row]) and self.shape_order_from_system_matrix(row) > 1):
                raise PropagatorGenerationException(
                    "For symbol " + str(self.x_[row]) + ": higher-order inhomogeneous ODEs are not supported")

            update_expr_terms = []
            for col in range(P.shape[1]):
                if not _is_zero(
                        P[row, col]):  # for every nonzero entry of the propagator matrix name the symbol
                    sym_str = (Config().propagators_prefix + "__{}__{}".format(str(self.x_[row]), str(self.x_[col])))
                    P_expr[sym_str] = P[row, col]  # store the value

                    if col in particular_solutions:  # if col contains a known solution after propagations the deviation from steady state forward
                        update_expr_terms.append(
                            sym_str + " * (" + str(self.x_[col]) + " - (" + str(particular_solutions[col]) + "))")
                    else:
                        # no solution, isolated solving
                        update_expr_terms.append(
                            sym_str + " * " + str(self.x_[col]))

            if row in particular_solutions:
                # add back this row's own steady-state offset
                update_expr_terms.append(
                    "(" + str(particular_solutions[row]) + ")")

            # handle non-invertible matrices by implementing linear drift term
            elif row in constant_drift_rows:
                update_expr_terms.append(
                    Config().output_timestep_symbol + " * (" + str(self.b_[row]) + ")")

            # parses solution from plain py into sympy
            update_expr[str(self.x_[row])] = " + ".join(update_expr_terms)
            update_expr[str(self.x_[row])] = _sympy_parse_real(
                update_expr[str(self.x_[row])], global_dict=Shape._sympy_globals)

            if not _is_zero(self.b_[row]):
                update_expr[str(self.x_[row])] = (_custom_simplify_expr(
                    update_expr[str(self.x_[row])]))  # simplify expression

        # gather and store propagators
        all_state_symbols = [str(sym) for sym in self.x_]
        initial_values = {sym: str(self.get_initial_value(sym))
                          for sym in all_state_symbols}
   

    def _get_coupled_components_(self):
        """
        Find groups of state variables that are coupled through the system matrix. 

        Two state variables belong to the same component if they are connected
        through a non-zero coefficient in `A`. The coupling is treated as
        undirectional and variables are treated together when computing a particular solution 
        """

        # intialise empty adjacency matrix A with 0s
        connectivity = np.zeros(self.A_.shape, dtype=int)

        for row in range(self.A_.shape[0]):    # for each row and column in A_ 
            for col in range(self.A_.shape[1]):

                if (not _is_zero(self.A_[row, col]) or not _is_zero(self.A_[col, row])):
                    connectivity[row, col] = 1   # if a unidirectional connection is found in the nest loop mark it as 1.

        # finds clusters of connected variables (1's; coupled blocks) converting a dense connectivity into a sparse
        _, component_labels = scipy.sparse.csgraph.connected_components(scipy.sparse.csr_matrix(connectivity), directed=False)

        # build a final list of blocks each inner nested block is coupled together
        return [ [i for i, label in enumerate(component_labels) if label == component] for component in set(component_labels) ]

    def generate_solver_dict_based_on_propagator_matrix_(self, P: sympy.Matrix):
        """
        This function generates the analytical solver dictionary representation based off the propagator matrix to advance the model forward at dt. 
        For isolated components, it just adds a fixed constant per dt (solving refr_T bug in #107). For coupled components, it measures the drift away from
        dt and calculates trajectory (x(t) - steady_state) for next dt. 
        """

        #
        # Generate symbols for each non-zero entry of the propagator matrix.
        #
        P_expr = {}
        update_expr = {}

        # Group coupled state variables so that their particular solutions
        # can be computed together.
        components = self._get_coupled_components_()

        # calculated steady-state equillibrium formula for all inhomogenous systems 
        particular_solutions = {}

        # Tracking varaibles that fit the drift profile (x' = b)
        constant_drift_rows = set()

        for indices in components:
            A_block = self.A_.extract(indices, indices)   # for each coupled group, extract submatrix for this block
            b_block = self.b_.extract(indices, [0])       # pulls out coresponding inhomogenous forcing terms 

            if all(_is_zero(b_block[i, 0]) for i in range(len(indices))):  # if the forcing terms are 0, the block is homogenous (non-linear)
                continue

            if len(indices) == 1 and _is_zero(A_block[0, 0]):    # if there is a driving force, but coupling is 0,0 means the ode is x'=b (constant drift)
                constant_drift_rows.add(indices[0])
                continue

            try:    # for everyother case, solve steady state algebraically through matrix inversion (finding identity matrix) 
                x_particular = -(A_block.inv() * b_block)     # solving the entire blocks steady state 

            except NonInvertibleMatrixError as exc:      # if ablock is non-invertible meaning no steady state 
                raise PropagatorGenerationException(
                    "Could not compute a particular solution / steady state for the coupled "
                    "inhomogeneous system containing: " + ", ".join(str(self.x_[i]) for i in indices)) from exc

            for local_idx, global_idx in enumerate(indices):   # simplify and store the particular solution with helper function 
                particular_solutions[global_idx] = _custom_simplify_expr(
                    x_particular[local_idx, 0])

        for row in range(P.shape[0]):  # build the update expressions from P 
            if not _is_zero(self.c_[row]):    # iterate row-by-row over p matrix per state variable, old code kept the same 
                raise PropagatorGenerationException(
                    "For symbol " + str(self.x_[row]) + ": nonlinear part should be zero for propagators")    # split eq Ax, B, C expect linearity/non-linearity 

            # Higher-order inhomogeneous equations are not supported.
            if (not _is_zero(self.b_[row]) and self.shape_order_from_system_matrix(row) > 1):
                raise PropagatorGenerationException(
                    "For symbol " + str(self.x_[row]) + ": higher-order inhomogeneous ODEs are not supported")

            update_expr_terms = []

            for col in range(P.shape[1]):
                if _is_zero(P[row, col]):
                    continue

                # Create a symbol for each non-zero propagator entry.
                sym_str = (
                    Config().propagators_prefix + "__{}__{}".format(str(self.x_[row]), str(self.x_[col])))

                P_expr[sym_str] = P[row, col]

                if col in particular_solutions:    # Propagate the state relative to its particular solution
                    update_expr_terms.append(      # calculate the mathematical shift for coupled inhomogenous equations 
                        sym_str + " * (" + str(self.x_[col]) + " - (" + str(particular_solutions[col]) + "))")   # shift from steady state to actual value 
                else:
                    update_expr_terms.append( 
                        sym_str + " * " + str(self.x_[col]))    

            if row in particular_solutions:
                update_expr_terms.append(      # Add the particular solution back after propagation to derive x final 
                    "(" + str(particular_solutions[row]) + ")")

            elif row in constant_drift_rows:   # handle equation if there isn't a particular solution 
                update_expr_terms.append(      # special case x' = b of an isolated component 
                    Config().output_timestep_symbol + " * (" + str(self.b_[row]) + ")") 

            # combine string components and parse them for sympy 
            update_expr[str(self.x_[row])] = " + ".join(update_expr_terms)
            update_expr[str(self.x_[row])] = _sympy_parse_real(
                update_expr[str(self.x_[row])],
                global_dict=Shape._sympy_globals)

            if not _is_zero(self.b_[row]):
                update_expr[str(self.x_[row])] = _custom_simplify_expr(
                    update_expr[str(self.x_[row])])

        # construct final solver dictionary blueprint 
        all_state_symbols = [str(sym) for sym in self.x_]
        initial_values = {
            sym: str(self.get_initial_value(sym))
            for sym in all_state_symbols}

        solver_dict = {    # final assembly for the analytical solver dict 
            "solver": "analytical",
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
        initial_values = {sym: str(self.get_initial_value(sym))
                          for sym in all_state_symbols}

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
                    update_expr_terms.append(
                        str(y) + " * (" + str(self.A_[row, col]) + ")")
            update_expr[str(x)] = " + ".join(update_expr_terms) + \
                " + (" + str(self.b_[row]) + ") + (" + str(self.c_[row]) + ")"
            update_expr[str(x)] = _sympy_parse_real(
                update_expr[str(x)], global_dict=Shape._sympy_globals)

        # custom expression simplification
        for name, expr in update_expr.items():
            update_expr[name] = _custom_simplify_expr(expr)
            collect_syms = [sym for sym in update_expr[name].free_symbols if not (
                sym in state_variables or str(sym) in state_variables)]
            update_expr[name] = sympy.collect(update_expr[name], collect_syms)

        return update_expr

    def shape_order_from_system_matrix(self, idx: int) -> int:
        r"""Determine shape differential order from system matrix of symbol ``self.x_[idx]``"""
        N = self.A_.shape[0]
        A = np.zeros((N, N), dtype=int)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] = not _is_zero(self.A_[i, j])

        scc = scipy.sparse.csgraph.connected_components(
            A, connection="strong")[1]
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

        scc = scipy.sparse.csgraph.connected_components(
            A, connection="strong")[1]
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
                x[i] = shape.get_state_variables(
                    derivative_symbol=Config().differential_order_symbol)[j]
                i += 1

        i = 0
        for shape in shapes:
            highest_diff_sym_idx = [k for k, el in enumerate(x) if el == sympy.Symbol(str(
                shape.symbol) + Config().differential_order_symbol * (shape.order - 1), real=True)][0]
            shape_expr = shape.reconstitute_expr()

            #
            #   grab the defining expression and separate into linear and nonlinear part
            #

            lin_factors, inhom_term, nonlin_term = Shape.split_lin_inhom_nonlin(
                shape_expr, x, parameters=parameters)
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
