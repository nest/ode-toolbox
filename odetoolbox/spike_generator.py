import io
import math
import numpy as np
import random

    
class SpikeGenerator():
    
    @classmethod
    def spike_times_from_json(cls, stimuli, sim_time, derivative_symbol="__d"):
        """Read or generate spike times according to a JSON specification
        
        Returns
        -------
        spike_times : Dict[str -> List[float]]
            Each variable symbol is a key in the dictionary, and the list of spike times is the corresponding value. Symbol names use `derivative_symbol` to indicate differential order.
        """

        spike_times = {}
        for stimulus in stimuli:
            for sym in set(stimulus["variables"]):
                assert type(sym) is str
                sym = sym.replace("'", derivative_symbol)
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
    def _generate_homogeneous_poisson_spikes(cls, T, rate, min_isi=1E-6):
        """Generate spike trains for the given simulation length. Uses a Poisson distribution to create biologically realistic characteristics of the spike-trains.

        Parameters
        ----------
        T : float
            Spikes are generated in the window [0, T]. T is in s.
        min_isi : float
            Minimum time between two consecutive spikes, in s.

        Returns
        -------
        spike_times : dict(str -> list of float)
            For each symbol: a list with spike times
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
    def _generate_regular_spikes(cls, T, rate):
        """Generate spike trains for the given simulation length.

        Parameters
        ----------
        T : float
            Spikes are generated in the window (0, T]. T is in s.

        Returns
        -------
        spike_times : dict(str -> list of float)
            For each symbol: a list with spike times
        """

        spike_times = []
        isi = 1 / rate
        t = 0.
        while t < T:
            t += isi
            if t <= T:
                spike_times.append(t)

        return spike_times
