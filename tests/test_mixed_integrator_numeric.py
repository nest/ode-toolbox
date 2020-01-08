#
# test_mixed_integrator_numeric.py
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
import pytest
import unittest
import sympy
import numpy as np
#np.seterr(under="warn")

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except:
    INTEGRATION_TEST_DEBUG_PLOTS = False


from .context import odetoolbox
from odetoolbox.mixed_integrator import MixedIntegrator

from math import e
from sympy import exp, sympify
import sympy.parsing.sympy_parser

import scipy
import scipy.special
import scipy.linalg
from scipy.integrate import solve_ivp


try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError as ie:
    PYGSL_AVAILABLE = False


def open_json(fname):
    absfname = os.path.join(os.path.abspath(os.path.dirname(__file__)), fname)
    with open(absfname) as infile:
        indict = json.load(infile)
    return indict


class TestMixedIntegrationNumeric(unittest.TestCase):
    '''Numerical validation of MixedIntegrator. Note that this test uses all-numeric (no analytic part) integration to test for time grid aliasing effects of spike times.

    Simulate a conductance-based integrate-and-fire neuron which is receiving spikes. Check for a match of the final system state with a numerical reference value that was validated by hand.
    '''

    @pytest.mark.skipif(not PYGSL_AVAILABLE, reason="Need GSL integrator to perform test")
    def test_mixed_integrator_numeric(self):
        debug = True

        h = 1E-3    # [s]
        T = 5E-3    # [s]

        # neuron parameters
        tau = 20E-3    # [s]
        tau_syn = 5E-3    # [s]
        c_m = 1E-6    # [F]

        initial_values = { "V_m" : -82.123E-3, "g_in" : 8.9E-9, "g_in__d" : .43, "g_ex" : 2.4E-9, "g_ex__d" : .06 }
        spike_times = { "g_ex__d" : np.array([1E-3, 2E-3, 3E-3]), "g_in__d" : np.array([.5E-3, 1.5E-3, 2.5E-3]) }

        ###

        N = int(np.ceil(T / h) + 1)
        timevec = np.linspace(0., T, N)

        initial_values = { sympy.Symbol(k) : v for k, v in initial_values.items() }

        indict = open_json("iaf_cond_alpha_mixed_test.json")
        analysis_json, shape_sys, shapes = odetoolbox.analysis_(indict, enable_stiffness_check=False, disable_analytic_solver=True)
        print("Got analysis result from ode-toolbox: ")
        print(json.dumps(analysis_json, indent=2))
        assert len(analysis_json) == 1
        assert analysis_json[0]["solver"].startswith("numeric")

        for alias_spikes in [False, True]:
            for integrator in [odeiv.step_rk4, odeiv.step_bsimp]:
                mixed_integrator = MixedIntegrator(
                 integrator,
                 shape_sys,
                 shapes,
                 analytic_solver_dict=None, #analysis_json[0],
                 parameters=indict["parameters"],
                 spike_times=spike_times,
                 random_seed=123,
                 max_step_size=h,
                 integration_accuracy_abs=1E-5,
                 integration_accuracy_rel=1E-5,
                 sim_time=T,
                 alias_spikes=alias_spikes)
                h_min, h_avg, runtime, upper_bound_crossed, t_log, h_log, y_log, sym_list = mixed_integrator.integrate_ode(
                 initial_values=initial_values,
                 h_min_lower_bound=1E-12, raise_errors=True, debug=True) # debug needs to be True here to obtain the right return values

                if INTEGRATION_TEST_DEBUG_PLOTS:
                    self._timeseries_plot(
                     t_log,
                     h_log,
                     y_log,
                     sym_list=sym_list,
                     basedir="/tmp",
                     fn_snip="_[alias=" + str(alias_spikes) + "]_" + str(integrator),
                     title_snip=" alias spikes: " + str(alias_spikes) + ", " + str(integrator))

                if alias_spikes:
                    assert upper_bound_crossed
                else:
                    assert not upper_bound_crossed


    def _timeseries_plot(self, t_log, h_log, y_log, sym_list, basedir="/tmp", fn_snip="", title_snip=""):
        if 1:
            fig, ax = plt.subplots(len(y_log[0]), sharex=True)
            for i, sym in enumerate(sym_list):
                ax[i].plot(1E3 * np.array(t_log), np.array(y_log)[:, i], label=str(sym))

            for _ax in ax:
                _ax.legend()
                _ax.grid(True)
                #_ax.set_xlim(49., 55.)

            ax[-1].set_xlabel("Time [ms]")
            fig.suptitle("Timeseries for mixed integrator numeric test" + title_snip)

            #plt.show()
            fn = os.path.join(basedir, "test_mixed_integrator_numeric_" + fn_snip + ".png")
            print("Saving to " + fn)
            plt.savefig(fn, dpi=600)
            plt.close(fig)

if __name__ == '__main__':
    unittest.main()
