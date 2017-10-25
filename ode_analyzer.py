#!/usr/bin/env python

from __future__ import print_function

import datetime
import json
import os
import sys

from sympy import diff, simplify
from sympy.parsing.sympy_parser import parse_expr

from analytic import compute_analytical_solution
from numeric import compute_numeric_solution
from shapes import shape_from_function, shape_from_ode


def ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes):
    """
    :param ode_symbol string encoding the LHS
    :param ode_definition string encoding RHS
    :param shapes A list with `Shape`-obejects
    :return true iff the ode definition is a linear and constant coefficient ODE
    """

    ode_symbol_sp = parse_expr(ode_symbol)
    ode_definition_sp = parse_expr(ode_definition)

    # check linearity
    ddvar = diff(diff(ode_definition_sp, ode_symbol_sp), ode_symbol_sp)

    if simplify(ddvar) != 0:
        return False

    # check coefficients
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


def main(args):
    """
    The main entry point. The main function expects a userdefined path a file passed throug the `sys.arv`
    :return: Stores its results in file with .json extension
    """
    print("Reading input file...")

    num_args = len(args)
    if num_args != 1:
        print("Wrong number of arguments (%d given, one expected)" % num_args)
        print("Usage: ode_analyzer <json_file>")
        print("Aborting.")
        sys.exit(exitcodes["wrong_num_args"])

    fname = args[0]

    if not os.path.isfile(fname):
        print("The file '%s' does not exist." % fname)
        print("Usage: ode_analyzer <json_file>")
        print("Aborting.")
        sys.exit(exitcodes["file_not_found"])

    with open(fname) as infile:
        try:
            input = json.load(infile)
        except Exception as e:
            print("The input JSON file could not be parsed.")
            print("Error: " + e.message)
            print("Please consult the file doc/example.json for help.")
            print("Aborting.")
            sys.exit(exitcodes["invalid_json_input"])

    print("Validating JSON...")

    for key in ["odes", "shapes"]:
        if not input.has_key(key):
            print("The key '%s' is not contained in the input file." % key)
            print("Please consult the file doc/example.json for help.")
            print("Aborting.")
            sys.exit(exitcodes["malformed_input"])

    print("Analyzing shapes...")
    shapes = []
    for shape in input["shapes"]:
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
    for ode in input["odes"]:
        print("  " + ode["symbol"], end="")
        lin_const_coeff = ode_is_lin_const_coeff(ode["symbol"], ode["definition"], shapes)
        ode["is_linear_constant_coefficient"] = lin_const_coeff
        if lin_const_coeff:
            prefix = " is a "
        else:
            prefix = " is no "
        print(prefix + "linear constant coefficient ODE.")

    print("Generating solvers...")

    for ode in input["odes"]:
        print("  " + ode["symbol"], end="")
        if ode["is_linear_constant_coefficient"]:
            print(": analytical")
            result = compute_analytical_solution(ode["symbol"], ode["definition"], shapes)
        else:
            print(": numerical ", end="")
            result = compute_numeric_solution(shapes)
            if "parameters" in input:
                try:
                    import pygsl
                    from stiffness import check_ode_system_for_stiffness
                    # prepare the original JSON for the testing. E.g. all shapes must be an ode with initial values
                    ode_shapes = []
                    for shape in shapes:
                        ode_shape = {"type": "ode",
                                     "symbol": str(shape.symbol),
                                     "initial_values": [str(x) for x in shape.initial_values],
                                     "definition": str(shape.ode_definition)}

                        ode_shapes.append(ode_shape)
                    input["shapes"] = ode_shapes
                    solver_type = check_ode_system_for_stiffness(input)
                    if solver_type == "implicit":
                        print("implicit scheme")
                    else:
                        print("explicit scheme")

                    result["evaluator"] = solver_type
                except ImportError:
                    result["evaluator"] = "skipped"
                    print("Please, install PyGSL in order to enable checking of the stiffness.")
            else:
                print("Please, provide `parameters` entry in the JSON to enable the stiffness check.")
                result["evaluator"] = "skipped"
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    result = main(sys.argv[1:])

    print("Writing output...")
    # TODO: hm, the naming scheme seems to me kind of arbitrary. why date? why not something
    # more reliable or more informative (e.g. with millisecond)?
    date = datetime.datetime.today().strftime('%Y%m%d')
    fname = "result-%s.json" % date
    print("  filename: %s" % fname)
    with open(fname, 'w') as outfile:
        outfile.write(result)

    print("Done.")
