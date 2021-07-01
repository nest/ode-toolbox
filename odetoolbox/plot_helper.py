#
# plot_helper.py
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

import logging

_mpl = None
_plt = None


def import_matplotlib():
    """Try to import and configure matplotlib. Returns the "mpl" and "plt" packages if the import was successful, or return (None, None) if unsuccessful."""
    global _mpl
    global _plt
    if _mpl:
        return _mpl, _plt
    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        def update_matplotlib_log_level():
            log_level = "WARNING"
            logging.getLogger("matplotlib.colorbar").setLevel(log_level)
            logging.getLogger("matplotlib.font_manager").setLevel(log_level)
            logging.getLogger("matplotlib.ticker").setLevel(log_level)

        update_matplotlib_log_level()
        _mpl, _plt = mpl, plt
        return _mpl, _plt

    except ImportError:
        return None, None
