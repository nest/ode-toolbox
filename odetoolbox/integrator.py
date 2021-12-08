#
# integrator.py
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

from typing import Dict, List, Optional

import numpy as np
import sympy
import sympy.matrices


class Integrator:
    r"""
    Integrate a dynamical system by means of the propagators returned by ODE-toolbox (base class).
    """

    all_variable_symbols = []   # type: List[sympy.Symbol]

    def set_spike_times(self, spike_times: Optional[Dict[str, List[float]]]):
        r"""
        Internally converts to a global, sorted list of spike times.

        :param spike_times: For each variable, used as a key, the list of spike times associated with it.
        """
        if spike_times is None:
            self.spike_times = {}
        else:
            self.spike_times = spike_times.copy()
        assert all([type(sym) is str for sym in self.spike_times.keys()]), "Spike time keys need to be of type str"
        self.all_spike_times = []  # type: List[float]
        self.all_spike_times_sym = []   # type: List[List[str]]
        for sym, sym_spike_times in self.spike_times.items():
            assert type(sym) is str
            assert str(sym) in [str(_sym) for _sym in self.all_variable_symbols], "Tried to set a spike time of unknown symbol \"" + sym + "\""
            for t_sp in sym_spike_times:
                if t_sp in self.all_spike_times:
                    idx = self.all_spike_times.index(t_sp)
                    self.all_spike_times_sym[idx].extend([sym])
                else:
                    self.all_spike_times.append(t_sp)
                    self.all_spike_times_sym.append([sym])

        idx = np.argsort(self.all_spike_times)
        self.all_spike_times = [self.all_spike_times[i] for i in idx]
        self.all_spike_times_sym = [self.all_spike_times_sym[i] for i in idx]


    def get_spike_times(self):
        r"""
        Get spike times.

        :return spike_times: For each variable, used as a key, the list of spike times associated with it.
        """
        return self.spike_times


    def get_sorted_spike_times(self):
        r"""
        Returns a global, sorted list of spike times.

        :return all_spike_times: A sorted list of all spike times for all variables.
        :return all_spike_times_sym: For the spike at time :python:`all_spike_times[i]`, the variables to which that spike applies are listed in :python:`all_spike_times_sym[i]`.
        """
        return self.all_spike_times, self.all_spike_times_sym
