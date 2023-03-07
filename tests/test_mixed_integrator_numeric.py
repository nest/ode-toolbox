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

import os
import pytest
import sympy
import numpy as np

try:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except ImportError:
    INTEGRATION_TEST_DEBUG_PLOTS = False

try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False

import odetoolbox
from odetoolbox.mixed_integrator import MixedIntegrator
from tests.test_utils import _open_json


def _timeseries_plot(t_log, h_log, y_log, sym_list, basedir="/tmp", fn_snip="", title_snip=""):
    fig, ax = plt.subplots(len(y_log[0]), sharex=True)
    for i, sym in enumerate(sym_list):
        ax[i].plot(1E3 * np.array(t_log), np.array(y_log)[:, i], label=str(sym))

    for _ax in ax:
        _ax.legend()
        _ax.grid(True)

    ax[-1].set_xlabel("Time [ms]")
    fig.suptitle("Timeseries for mixed integrator numeric test" + title_snip)

    fn = os.path.join(basedir, "test_mixed_integrator_numeric_" + fn_snip + ".png")
    print("Saving to " + fn)
    plt.savefig(fn, dpi=600)
    plt.close(fig)


def _run_simulation(indict, alias_spikes, integrator, params=None, **kwargs):
    """
    Parameters
    ----------
    params : Optional[Dict]
        Parameter values to pass to the integrator (overrides parameter values in the input json file).
    kwargs : Dict
        Extra parameters passed to ``odetoolbox.analysis()``.
    """
    h = 5E-3   # very big to see spike aliasing effects better [s]
    T = 50E-3  # [s]

    initial_values = {"g_ex__d": 0., "g_in__d": 0.}    # optionally override initial values
    initial_values = {sympy.Symbol(k): v for k, v in initial_values.items()}
    spike_times = {"g_ex__d": np.array([10E-3]), "g_in__d": np.array([6E-3])}

    analysis_json, shape_sys, shapes = odetoolbox._analysis(indict, disable_stiffness_check=True, disable_analytic_solver=True, log_level="DEBUG", **kwargs)

    assert len(analysis_json) == 1
    assert analysis_json[0]["solver"].startswith("numeric")

    _params = indict["parameters"]
    if params is not None:
        _params.update(params)

    debug_plot_dir = None
    if INTEGRATION_TEST_DEBUG_PLOTS:   # only enable plotting if matplotlib was successfully imported
        debug_plot_dir = "/tmp"

    mixed_integrator = MixedIntegrator(integrator,
                                       shape_sys,
                                       shapes,
                                       analytic_solver_dict=None,
                                       parameters=_params,
                                       spike_times=spike_times,
                                       random_seed=123,
                                       max_step_size=h,
                                       integration_accuracy_abs=1E-6,
                                       integration_accuracy_rel=1E-6,
                                       sim_time=T,
                                       alias_spikes=alias_spikes,
                                       debug_plot_dir=debug_plot_dir)
    h_min, h_avg, runtime, upper_bound_crossed, t_log, h_log, y_log, sym_list = \
        mixed_integrator.integrate_ode(initial_values=initial_values,
                                       h_min_lower_bound=1E-12,
                                       raise_errors=True,
                                       debug=True)		# debug needs to be True here to obtain the right return values
    return h_min, h_avg, runtime, upper_bound_crossed, t_log, h_log, y_log, sym_list, analysis_json


@pytest.mark.skipif(not PYGSL_AVAILABLE, reason="Need GSL integrator to perform test")
def test_mixed_integrator_numeric(**kwargs):
    """
    Numerical validation of MixedIntegrator. Note that this test uses all-numeric (no analytic part) integration to test for time grid aliasing effects of spike times.

    Simulate a conductance-based integrate-and-fire neuron which is receiving spikes. Check for a match of the final system state with a numerical reference value that was validated by hand.
    """

    integrator = odeiv.step_rk4

    for alias_spikes in [True, False]:
        indict = _open_json("iaf_cond_alpha.json")
        h_min, h_avg, runtime, upper_bound_crossed, t_log, h_log, y_log, sym_list, analysis_json = _run_simulation(indict, alias_spikes, integrator)

        if INTEGRATION_TEST_DEBUG_PLOTS:
            _timeseries_plot(t_log,
                             h_log,
                             y_log,
                             sym_list=sym_list,
                             basedir="/tmp",
                             fn_snip="_[alias=" + str(alias_spikes) + "]_" + str(integrator),
                             title_snip=" alias spikes: " + str(alias_spikes) + ", " + str(integrator))

        if alias_spikes:
            assert not upper_bound_crossed
        else:
            assert upper_bound_crossed
