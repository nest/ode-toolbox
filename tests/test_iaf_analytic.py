#
# test_iaf_analytic.py
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

import json
import os
import unittest
import sympy
import numpy as np

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except Exception:
    INTEGRATION_TEST_DEBUG_PLOTS = False

import odetoolbox
from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.spike_generator import SpikeGenerator
from math import e
from sympy import exp, sympify
import scipy
import scipy.special
import scipy.linalg


def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestAnalyticIntegratorIAF(unittest.TestCase):
    """
    Test that analytic integrator returns the same result when caching is disabled and enabled.
    """

    def test_analytic_integrator_iaf(self):
        debug = True
        h = 1  # [ms]
        T = 1000  # [ms]
        indict = open_json("iaf.json")
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True)
        print("Got solver_dict from ode-toolbox: ")
        print(json.dumps(solver_dict, indent=2))
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"
        ODE_INITIAL_VALUES = {"H_s": 1, "I_s": 0, "V_m": 0}
        _parms = {"C": 1, "tau_s": 10, "tau_m": 10}

        if not "parameters" in solver_dict.keys():
            solver_dict["parameters"] = {}
        solver_dict["parameters"].update(_parms)
        N = int(np.ceil(T / h) + 1)
        timevec = np.linspace(0., T, N)
        state = {True: {}, False: {}}
        state[True] = {sym: [] for sym in solver_dict["state_variables"]}
        state[True]["timevec"] = []
        analytic_integrator = AnalyticIntegrator(solver_dict, spike_times=None, enable_caching=True)  # spike_times, enable_caching=use_caching)
        analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)
        analytic_integrator.reset()
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)
            state[True]["timevec"].append(t)
            for sym, val in state_.items():
                state[True][sym].append(val)
        for k, v in state[True].items():
            state[True][k] = np.array(v)

        if INTEGRATION_TEST_DEBUG_PLOTS:
            fig, ax = plt.subplots(3, sharex=True)
            ax[0].plot(timevec, state[True]["H_s"], label="H_s")
            ax[1].plot(timevec, state[True]["I_s"], label="I_s")
            ax[2].plot(timevec, state[True]["V_m"], label="V_m")
            for _ax in ax:
                _ax.legend()
                _ax.grid(True)
            ax[-1].set_xlabel("Time [ms]")
            fn = os.path.join("", "test_iaf_analytic.png")
            print("Saving to " + fn)
            plt.savefig(fn, dpi=600)
            plt.close(fig)
        np.testing.assert_allclose(state[True]["timevec"], timevec)
        np.testing.assert_allclose(state[True]["timevec"], state[False]["timevec"])
        for sym, val in state_.items():
            np.testing.assert_allclose(state[True][sym], state[False][sym])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
