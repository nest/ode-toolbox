#
# numeric.py
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

def compute_numeric_solution(shapes):

    data = {
        "solver": "numeric",
        "shape_initial_values": {},
        "shape_ode_definitions": {}
    }

    for shape in shapes:
        data["shape_initial_values"][str(shape.symbol)] = {var_name : str(expr) for var_name, expr in shape.initial_values.items()}
        data["shape_ode_definitions"][str(shape.symbol)] = str(shape.ode_definition)

    return data
