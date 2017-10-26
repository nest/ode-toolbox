#!/usr/bin/env python
#
# ode_analyzer.py
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

import json
import os
import sys

from sympy import diff, simplify
from sympy.parsing.sympy_parser import parse_expr

from analytic import compute_analytical_solution
from numeric import compute_numeric_solution
from shapes import shape_from_function, shape_from_ode

try:
    from stiffness import check_ode_system_for_stiffness
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


exitcodes = {
    "wrong_num_args": 1,
    "file_not_found": 2,
    "invalid_json_input": 5,
    "malformed_input": 10,
    "shape_not_lin_hom_": 15,
}


def analysis(indict):
    """The main entry point of the analysis.

    This function expects a single dictionary with the keys `odes`,
    `parameters` and `shapes` that describe the input to the analysis.

    The exact format of the input entries is described in the file
    `README.md`.

    :return: The result of the analysis, again as a dictionary.

    """

    print("Validating input...")
    for key in ["odes", "parameters", "shapes"]:
        if not indict.has_key(key):
            print("The key '%s' is not contained in the input." % key)
            print("Please consult the file README.md for help.")
            print("Aborting.")
            sys.exit(exitcodes["malformed_input"])

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
            print("")
            print("The shape does not obey a linear homogeneous ODE.")
            print("Please check the definition of shape '%s'" % shape["symbol"])
            print("Aborting.")
            sys.exit(exitcodes["shape_not_lin_hom"])

    print("Analyzing ODEs...")
    for ode in indict["odes"]:
        print("  " + ode["symbol"], end="")
        lin_const_coeff = ode_is_lin_const_coeff(ode["symbol"], ode["definition"], shapes)
        ode["is_linear_constant_coefficient"] = lin_const_coeff
        prefix = " is a " if lin_const_coeff else " is no "
        print(prefix + "linear constant coefficient ODE.")

    print("Generating solvers...")

    for ode in indict["odes"]:
        print("  " + ode["symbol"], end="")
        if ode["is_linear_constant_coefficient"]:
            print(": analytical")
            output = compute_analytical_solution(ode["symbol"], ode["definition"], shapes)
        else:
            print(": numerical ", end="")
            output = compute_numeric_solution(shapes)
            if HAVE_STIFFNESS:
                ode_shapes = []
                for shape in shapes:
                    ode_shape = {"type": "ode",
                                 "symbol": str(shape.symbol),
                                 "initial_values": [str(x) for x in shape.initial_values],
                                 "definition": str(shape.ode_definition)}
                    ode_shapes.append(ode_shape)
                indict["shapes"] = ode_shapes
                solver_type = check_ode_system_for_stiffness(indict)
                output["solver"] += "-" + solver_type
                print(solver_type + " scheme")
            else:
                print("(stiffness test skipped, PyGSL not available)")

    return output


if __name__ == "__main__":

    args = sys.argv[1:]

    print("Reading input file...")

    num_args = len(args)
    if num_args != 1:
        print("Wrong number of arguments (%d given, one expected)" % num_args)
        print("Usage: ode_analyzer <json_file>")
        print("Aborting.")
        sys.exit(exitcodes["wrong_num_args"])

    infname = args[0]
    if not os.path.isfile(infname):
        print("The file '%s' does not exist." % infname)
        print("Usage: ode_analyzer <json_file>")
        print("Aborting.")
        sys.exit(exitcodes["file_not_found"])

    with open(infname) as infile:
        try:
            indict = json.load(infile)
        except Exception as e:
            print("The input JSON file could not be parsed.")
            print("Error: " + e.message)
            print("Please consult the file doc/example.json for help.")
            print("Aborting.")
            sys.exit(exitcodes["invalid_json_input"])

    result = analysis(indict)

    print("Writing output...")

    basename = os.path.basename(infname.rsplit(".", 1)[0])
    outfname = "%s_result.json" % basename
    print("  filename: %s" % outfname)
    with open(outfname, 'w') as outfile:
        outfile.write(json.dumps(result, indent=2))

    print("Done.")
