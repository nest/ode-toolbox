#!/usr/bin/env python

from __future__ import print_function

import datetime, sys, os
import json

from sympy import diff, simplify, Symbol, sympify
from sympy.parsing.sympy_parser import parse_expr

from exact import compute_exact_solution
from numeric import compute_numeric_solution
from shapes import shape_from_function, shape_from_ode


def ode_is_lin_const_coeff(ode_symbol, ode_definition, shapes):
    """
    """

    shape_symbols = {}
    for shape in shapes:
        shape_symbols[shape.symbol] = parse_expr(shape.ode_definition)

    ode_symbol_sp = parse_expr(ode_symbol)
    ode_definition_sp = parse_expr(ode_definition, local_dict=shape_symbols)

    dvar = diff(ode_definition_sp, ode_symbol_sp)
    dtdvar = diff(dvar, Symbol("t"))

    return simplify(dtdvar) == sympify(0)


exitcodes = {
    "wrong_num_args": 1,
    "file_not_found": 2,
    "invalid_json_input": 5,
    "malformed_input": 10,
    "shape_not_lin_hom_": 15,
}


if __name__ == "__main__":
    
    print("Reading input file...")

    num_args = len(sys.argv)
    if num_args != 2:
        print("Wrong number of arguments (%d given, one expected)" % num_args)
        print("Usage: ode_analyzer <json_file>")
        print("Aborting.")
        sys.exit(exitcodes["wrong_num_args"])        

    fname = sys.argv[1]

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

    for key in ["odes", "parameters", "shapes"]:
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
            print(": exact")
            result = compute_exact_solution(ode["symbol"], ode["definition"], shapes)
        else:
            print(": numerical")
            result = compute_numerical_solution(ode["symbol"], ode["definition"], shapes)
            #TODO: run the stiffness tester

    print(result)
            
    print("Writing output...")
    date = datetime.datetime.today().strftime('%Y%m%d')
    fname = "result-%s.json" % date
    print("  filename: %s" % fname)
    with open(fname, 'w') as outfile:
        outfile.write(result)

    print("Done.")
