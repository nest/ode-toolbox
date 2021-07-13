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

from typing import Iterable

import argparse
import json
import logging
import os
import sys

import odetoolbox
from odetoolbox.shapes import MalformedInputException
from odetoolbox import _init_logging


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="""ode-toolbox -- https://github.com/nest/ode-toolbox""", formatter_class=argparse.RawDescriptionHelpFormatter)

    argparser.add_argument("infile", metavar='PATH', type=str, help="JSON input file path")
    argparser.add_argument("--disable-stiffness-check", action="store_true", help="If provided, disable stiffness check")
    argparser.add_argument("--disable-analytic-solver", action="store_true", help="If provided, disable generation of propagators")
    argparser.add_argument("--preserve-expressions", action="store", nargs="*", default=False, help="Set to True, or a list of strings corresponding to individual variable names, to disable internal rewriting of expressions, and return same output as input expression where possible. Can only apply to variables specified as first-order differential equations.")
    argparser.add_argument("--simplify-expression", action="store", default="sympy.simplify(expr)", help="For all expressions ``expr`` that are rewritten internally: the contents of this parameter string are evaluated with ``eval()`` in Python to obtain the final output expression. Override for custom expression simplification steps. Example: ``\"sympy.logcombine(sympy.powsimp(sympy.expand(expr)))\"``.")
    argparser.add_argument("--log-level", action="store", default="WARN", help="Sets the logging threshold. Logging messages which are less severe than ``log_level`` will be ignored. Log levels can be provided as an integer or string, for example \"INFO\" (more messages) or \"WARN\" (fewer messages). For a list of valid logging levels, see https://docs.python.org/3/library/logging.html#logging-levels")
    parsed_args = argparser.parse_args()

    if isinstance(parsed_args.preserve_expressions, Iterable) and len(parsed_args.preserve_expressions) == 0:
        parsed_args.preserve_expressions = True

    _init_logging(log_level=parsed_args.log_level)
    logging.info("Reading input file...")

    if not os.path.isfile(parsed_args.infile):
        logging.error("The file '%s' does not exist." % parsed_args.infile)
        sys.exit(1)

    with open(parsed_args.infile) as infile:
        try:
            indict = json.load(infile)
        except Exception as e:
            logging.error("The input JSON file could not be parsed; error: " + e.msg)
            sys.exit(1)

    try:
        result = odetoolbox.analysis(indict,
                                     disable_stiffness_check=parsed_args.disable_stiffness_check,
                                     disable_analytic_solver=parsed_args.disable_analytic_solver,
                                     preserve_expressions=parsed_args.preserve_expressions,
                                     simplify_expression=parsed_args.simplify_expression,
                                     log_level=parsed_args.log_level)
    except MalformedInputException as e:
        logging.error("The input JSON file could not be parsed; error: " + e.message)
        sys.exit(1)

    basename = os.path.basename(parsed_args.infile.rsplit(".", 1)[0])
    outfname = "%s_result.json" % basename
    logging.info("Writing output to file %s..." % outfname)
    with open(outfname, 'w') as outfile:
        outfile.write(json.dumps(result, indent=2))
