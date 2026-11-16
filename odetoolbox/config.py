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

import copy


class Config:
    r"""Static class to store global configuration options.

    Options are stored in the static dict ``config``. Access using either :python:`Config().key` or :python:`Config()["key"]` or :python:`Config.config["key"]`.

    The options persist across invocations of ODE-toolbox in the same process; call :py:meth:`reset` to restore the default values.
    """

    _default_config = {
        "simplify_expression": "sympy.simplify(expr)",
        "expression_simplification_threshold": 2500,
        "input_time_symbol": "t",
        "output_timestep_symbol": "__h",
        "differential_order_symbol": "__d",
        "propagators_prefix": "__P",
        "sim_time": 100E-3,
        "max_step_size": 999.,
        "integration_accuracy_abs": 1E-6,
        "integration_accuracy_rel": 1E-6,
        "forbidden_names": ["oo", "zoo", "nan", "NaN", "__h"],
        "use_alternative_expM": False
    }

    config = copy.deepcopy(_default_config)

    @classmethod
    def reset(cls):
        r"""Restore all options to their default values."""
        cls.config = copy.deepcopy(cls._default_config)

    def __getitem__(self, key):
        return Config.config[key]

    def __getattr__(self, key):
        return Config.config[key]

    def keys(self):
        return Config.config.keys()
