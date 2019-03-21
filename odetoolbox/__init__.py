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

from __future__ import print_function

from sympy import diff, simplify
from sympy.parsing.sympy_parser import parse_expr

from . import analytic
from . import numeric
from . import shapes

try:
    from . import stiffness
    HAVE_STIFFNESS = True
except:
    HAVE_STIFFNESS = False


def ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes):
    """
    :param ode_symbol string encoding the LHS
    :param ode_definition string encoding RHS
    :param shapes A list with `Shape`-obejects
    :return true iff the ode definition is a linear and constant coefficient ODE
    """

    ode_symbol_sp = parse_expr(ode_symbol)
    ode_definition_sp = parse_expr(ode_definition)

    # Check linearity
    ddvar = diff(diff(ode_definition_sp, ode_symbol_sp), ode_symbol_sp)

    if simplify(ddvar) != 0:
        return False

    # Check coefficients
    dvar = diff(ode_definition_sp, ode_symbol_sp)

    for shape in shapes:
        for symbol in dvar.free_symbols:
            if str(shape.symbol) == str(symbol):
                return False
    return True


class MalformedInput(Exception): pass

class ShapeNotLinHom(Exception): pass


def analysis(indict, enable_stiffness_check=True):
    """The main entry point of the analysis.

    This function expects a single dictionary with the keys `odes`,
    `parameters` and `shapes` that describe the input to the analysis.

    The exact format of the input entries is described in the file
    `README.md`.

    :return: The result of the analysis, again as a dictionary.

    """

    from odetoolbox.analytic import compute_analytical_solution
    from odetoolbox.numeric import compute_numeric_solution
    from odetoolbox.shapes import shape_from_function, shape_from_ode

    print("Validating input...")
    for key in ["odes", "parameters", "shapes"]:
        if key not in indict:
            raise MalformedInput("The key '%s' is not contained in the input." % key)

    print("Analyzing shapes...")
    shapes = []
    for shape in indict["shapes"]:
        try:
            print("  " + shape["symbol"], end="")
            if shape["type"] == "ode":
                shapes.append(shape_from_ode(**shape))
            else:
                shapes.append(shape_from_function(**shape))
            print(" is a linear homogeneous ODE")
        except Exception as e:
            print()
            raise ShapeNotLinHom("The shape does not obey a linear homogeneous ODE.")

    if len(indict["odes"]) == 0:
        print("Only shapes provided. Return ODE representation with IV.")
        return compute_numeric_solution(shapes)

    print("Analyzing ODEs...")
    for ode in indict["odes"]:
        print("  " + ode["symbol"], end="")
        lin_const_coeff = ode_is_lin_const_coeff(ode["symbol"], ode["definition"], shapes)
        ode["is_linear_constant_coefficient"] = lin_const_coeff
        prefix = " is a " if lin_const_coeff else " is not a "
        print(prefix + "linear constant coefficient ODE.")

    print("Generating solvers...")

    for ode in indict["odes"]:
        print("  " + ode["symbol"], end="")
        if ode["is_linear_constant_coefficient"]:
            print(": analytical")
            if "timestep_symbol_name" in indict.keys():
                output = compute_analytical_solution(ode["symbol"], ode["definition"], shapes, timestep_symbol_name=indict["timestep_symbol_name"])
            else:
                output = compute_analytical_solution(ode["symbol"], ode["definition"], shapes)
        else:
            print(": numerical ", end="")
            output = compute_numeric_solution(shapes)
            if HAVE_STIFFNESS and enable_stiffness_check:

                # TODO: check what happens/has to happen for shapes
                # that already have a definition of either type
                # `function` or `ode`.

                indict["shapes"] = []
                for shape in shapes:
                    ode_shape = {"type": "ode",
                                 "symbol": str(shape.symbol),
                                 "initial_values": [str(x) for x in shape.initial_values],
                                 "definition": str(shape.ode_definition)}
                    indict["shapes"].append(ode_shape)

                tester = stiffness.StiffnessTester(indict)

                # TODO: Check whether we need to run this with
                # arguments different from the defaults.
                solver_type = tester.check_stiffness()

                output["solver"] += "-" + solver_type
                print(solver_type + " scheme")

            else:
                print("(stiffness test skipped, PyGSL not available)")

    return output
