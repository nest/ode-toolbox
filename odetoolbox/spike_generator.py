#
# spike_generator.py
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

from typing import Mapping, List

import io
import math
import numpy as np
import random

from .config import Config


class SpikeGenerator():

    @classmethod
    def spike_times_from_json(cls, stimuli, sim_time) -> Mapping[str, List[float]]:
        r"""
        Read or generate spike times according to a JSON specification.

        :return: spike_times: Each variable symbol is a key in the dictionary, and the list of spike times is the corresponding value. Symbol names use `derivative_symbol` to indicate differential order.
        """
        spike_times = {}
        for stimulus in stimuli:
            for sym in set(stimulus["variables"]):
                assert type(sym) is str
                sym = sym.replace("'", Config().differential_order_symbol)
                if not sym in spike_times.keys():
                    spike_times[sym] = []

                if stimulus["type"] == "poisson_generator":
                    spike_times[sym].extend(SpikeGenerator._generate_homogeneous_poisson_spikes(T=sim_time, rate=float(stimulus["rate"])))
                elif stimulus["type"] == "regular":
                    spike_times[sym].extend(SpikeGenerator._generate_regular_spikes(T=sim_time, rate=float(stimulus["rate"])))
                elif stimulus["type"] == "list":
                    str_io = io.StringIO(stimulus["list"])
                    spikes = np.loadtxt(str_io)
                    spikes = np.sort([t_sp for t_sp in spikes if t_sp <= sim_time])
                    spike_times[sym].extend(spikes)
                else:
                    assert False, "Unknown stimulus type: \"" + str(stimulus["type"]) + "\""

        return spike_times


    @classmethod
    def _generate_homogeneous_poisson_spikes(cls, T: float, rate: float, min_isi: float = 1E-6):
        r"""
        Generate spike trains for the given simulation length. Uses a Poisson distribution to create biologically realistic characteristics of the spike-trains.

        :param T: Spikes are generated in the window :math:`[0, T]`.
        :param min_isi: Minimum time between two consecutive spikes.

        :return: spike_times: For each symbol: a list with spike times
        :rtype: Dict(str -> List(float))
        """
        spike_times = []
        t = 0.
        while t < T:
            isi = -math.log(1. - random.random()) / rate
            isi = max(isi, min_isi)
            t += isi
            if t <= T:
                spike_times.append(t)

        return spike_times


    @classmethod
    def _generate_regular_spikes(cls, T: float, rate: float):
        r"""
        Generate spike trains for the given simulation length.

        :param T: Spikes are generated in the window :math:`\langle 0, T]`.

        :return: For each symbol: a list with spike times
        :rtype: Dict(str -> List(float))
        """
        spike_times = []
        isi = 1 / rate
        t = 0.
        while t < T:
            t += isi
            if t <= T:
                spike_times.append(t)

        return spike_times
