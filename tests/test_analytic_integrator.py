#
# test_analytic_integrator.py
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
except ImportError:
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


class TestAnalyticIntegrator(unittest.TestCase):
    """
    Test that analytic integrator returns the same result when caching is disabled and enabled.
    """

    def test_analytic_integrator_iaf_psc_alpha(self):
        debug = True

        h = 1E-3    # [s]
        T = 100E-3    # [s]


        #
        #   timeseries using ode-toolbox generated propagators
        #

        indict = open_json("test_analytic_integrator.json")
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True)
        print("Got solver_dict from ode-toolbox: ")
        print(json.dumps(solver_dict, indent=2))
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"] == "analytical"

        ODE_INITIAL_VALUES = {"I": 0., "I__d": 0.}

        _parms = {"Tau": 2E-3,    # [s]
                  "e": sympy.exp(1)}

        if not "parameters" in solver_dict.keys():
            solver_dict["parameters"] = {}
        solver_dict["parameters"].update(_parms)

        spike_times = SpikeGenerator.spike_times_from_json(indict["stimuli"], T)

        N = int(np.ceil(T / h) + 1)
        timevec = np.linspace(0., T, N)
        state = {True: {}, False: {}}
        for use_caching in [False, True]:
            state[use_caching] = {sym: [] for sym in solver_dict["state_variables"]}
            state[use_caching]["timevec"] = []
            analytic_integrator = AnalyticIntegrator(solver_dict, spike_times, enable_caching=use_caching)
            analytic_integrator.set_initial_values(ODE_INITIAL_VALUES)
            analytic_integrator.reset()
            for step, t in enumerate(timevec):
                print("Step " + str(step) + " of " + str(N))
                state_ = analytic_integrator.get_value(t)
                state[use_caching]["timevec"].append(t)
                for sym, val in state_.items():
                    state[use_caching][sym].append(val)

        for use_caching in [False, True]:
            for k, v in state[use_caching].items():
                state[use_caching][k] = np.array(v)

        if INTEGRATION_TEST_DEBUG_PLOTS:
            fig, ax = plt.subplots(2, sharex=True)

            ax[0].plot(1E3 * timevec, state[True]["I"], linewidth=2, linestyle='--', dashes=(5, 1), marker="x", label="I (caching)", alpha=.8)
            ax[0].plot(1E3 * timevec, state[False]["I"], linewidth=2, linestyle=":", marker="o", label="I", alpha=.8)
            ax[1].plot(1E3 * timevec, state[True]["I__d"], linewidth=2, linestyle='--', dashes=(5, 1), marker="x", label="I' (caching)", alpha=.8)
            ax[1].plot(1E3 * timevec, state[False]["I__d"], linewidth=2, linestyle=":", marker="o", label="I'", alpha=.8)

            for _ax in ax:
                _ax.legend()
                _ax.grid(True)

            ax[-1].set_xlabel("Time [ms]")
            fn = os.path.join("", "test_analytic_integrator.png")
            print("Saving to " + fn)
            plt.savefig(fn, dpi=600)
            plt.close(fig)
            """
            fn = "/tmp/test_analytic_integrator.png"
            print("Saving to " + fn)
            plt.savefig(fn, dpi=600)
            plt.close(fig)
            """

        np.testing.assert_allclose(state[True]["timevec"], timevec)
        np.testing.assert_allclose(state[True]["timevec"], state[False]["timevec"])
        for sym, val in state_.items():
            np.testing.assert_allclose(state[True][sym], state[False][sym])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
