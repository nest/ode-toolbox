#
# config.py
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

class Config:
    r"""Static class to store global configuration options.

    Options are stored in the static dict ``config``. Access using either :python:`Config().key` or :python:`Config()[key]` or :python:`Config.config[key]`.
    """

    config = {
        "simplify_expression": "sympy.simplify(expr)",
        "expression_simplification_threshold": 1000,
        "input_time_symbol": "t",
        "output_timestep_symbol": "__h",
        "differential_order_symbol": "__d",
        "sim_time": 100E-3,
        "max_step_size": 999.,
        "integration_accuracy_abs": 1E-6,
        "integration_accuracy_rel": 1E-6,
        "forbidden_names": ["oo", "zoo", "nan", "NaN", "__h"]
    }

    def __getitem__(self, key):
        return Config.config[key]

    def __getattr__(self, key):
        return Config.config[key]

    def keys(self):
        return Config.config.keys()
