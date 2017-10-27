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
import odetoolbox


exitcodes = {
    "wrong_num_args": 1,
    "file_not_found": 2,
    "invalid_json_input": 5,
    "malformed_input": 10,
    "shape_not_lin_hom_": 15,
}


if __name__ == "__main__":

    import json
    import os
    import sys

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

    try:
        result = odetoolbox.analysis(indict)
    except odetoolbox.MalformedInput as e:
        print(e.message)
        print("Please consult the file README.md for help.")
        print("Aborting.")
        sys.exit(exitcodes["malformed_input"])
    except odetoolbox.ShapeNotLinHom as e:
        print(e.message)
        print("Please check the definition of shape '%s'" % shape["symbol"])
        print("Aborting.")
        sys.exit(exitcodes["shape_not_lin_hom"])

    print("Writing output...")
    basename = os.path.basename(infname.rsplit(".", 1)[0])
    outfname = "%s_result.json" % basename
    print("  filename: %s" % outfname)
    with open(outfname, 'w') as outfile:
        outfile.write(json.dumps(result, indent=2))

    print("Done.")
