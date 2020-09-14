#
# test_fitzhughnagumo.py
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
from scipy.signal import find_peaks
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve
from sympy import Symbol
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except Exception:
    INTEGRATION_TEST_DEBUG_PLOTS = False
import odetoolbox
from odetoolbox.mixed_integrator import MixedIntegrator
from math import e
from sympy import exp, sympify
import sympy.parsing.sympy_parser
import scipy
import scipy.special
import scipy.linalg

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


class TestFitxhughNagumo(unittest.TestCase):
    """
    This is the FitzHugh-Nagumo model [http://www.scholarpedia.org/article/FitzHugh-Nagumo_model, https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model]
    Implementing the fitzhughNagumo model starting from equilibrium values, and performing a test that if the external current crosses a certain threshold value, regular spikes are obtained.
    This function tests if the number of spikes cross 20 in that case.
    Additionally, plots of V and W vs time are obtained for different values of current, and a FI curve is also plotted.
    """
    def initial__values(self, curr):
        """
        This function returns the initial values(for every  value of external current), i.e, the equilibrium values of V and W where the conditon dV/dt = dW/dt = 0 is staisfied.
        Hence, V and W are the roots of the following equations:
        V - V**3/3 - W + I = 0
        0.08*(V + 0.7 - 0.8*W) = 0
        Sympy is used for the calculation of the roots.
        """
        I_ext = Symbol("I_ext")
        V = Symbol("V")
        expr = solve((sympy.parsing.sympy_parser.parse_expr("8*V**3 + 6*V + 21 - 24*I_ext")), V)  # expr gives a list of three roots for V: first two are complex, third one is real
        final_val_V = (expr[2].subs(I_ext, curr)).evalf()
        final_val_W = ((10 * final_val_V) + 7) / 8
        return float(final_val_V), float(final_val_W)  # since sympy returns objects, we convert final_val_v and final_val_w to float numbers

    @pytest.mark.skipif(not PYGSL_AVAILABLE, reason="Need GSL integrator to perform test")
    def test_fitzhugh_nagumo(self):
        debug = True
        h = 1  # [ms] #time steps
        T = 1000  # [ms] #total simulation time
        n = 10  # total number of current values between 0 and 1
        I_ext = np.linspace(0, 1, n)  # external current
        small_perturb = 0.001  # this value is the slight disturbance that we introduce to the equilibrium value of V returned from the initial__values() function
        threshold_V_for_peak = 1.5  # the minimum value of V for it to be counted as a peak
        """
        Since about the first 200 ms correspond to a transient state of the neuron from exhibiting no spikes to gradually spiking (if the current is sufficient),
        We start our analysis after ignoring the initial 200 ms and count the peaks appearing in the rest of the simulation time. N1 is therefore the index of the starting time.
        """
        time_analysis_start = 200  # starting our ananlysis after 200 ms
        N1 = int(np.ceil(time_analysis_start / h))  # index of the starting time
        peak_freq = np.zeros(n)
        indict = open_json("fitzhughnagumo.json")
        analysis_json, shape_sys, shapes = odetoolbox._analysis(indict, disable_stiffness_check=True, disable_analytic_solver=True)
        print("Got analysis result from ode-toolbox: ")
        print(json.dumps(analysis_json, indent=2))
        assert len(analysis_json) == 1
        assert analysis_json[0]["solver"].startswith("numeric")
        integrator = odeiv.step_rk4
        for j in range(n):
            # loop over current values
            initial_values = {"V": (self.initial__values(I_ext[j])[0] + small_perturb), "W": self.initial__values(I_ext[j])[1]}
            initial_values = {sympy.Symbol(k): v for k, v in initial_values.items()}
            mixed_integrator = MixedIntegrator(integrator, shape_sys, shapes, analytic_solver_dict=None, parameters={"I_ext": str(I_ext[j])}, max_step_size=h, integration_accuracy_abs=1E-5, integration_accuracy_rel=1E-5, sim_time=T)
            h_min, h_avg, runtime, upper_bound_crossed, t_log, h_log, y_log, sym_list = mixed_integrator.integrate_ode(initial_values=initial_values, h_min_lower_bound=1E-12, raise_errors=True, debug=True)  # debug needs to be True here to obtain the right return values
            peak_freq[j] = self.peak_detection(y_log, N1, threshold_V_for_peak, time_analysis_start, T)
            if I_ext[j] > 1 / 3:  # this is actual unit testing part.
                """
                I = 0.333333..: In the plot we see that the system gradually gets to a state where it starts spiking regularly. Therefore this current can
                be regarded as the threshold current where the equilibrium shifts from a stable one to an unstable one. The threshold theoretically is (1/3)
                which is a non terminating number. However, the computer cannot store an infinitely long number, and hence it rounds up the numbers.
                One possibility is that (1/3) is rounded up to 0.33333333...4. Which is slightly above the threshold and hence we see that the system exhibits regular spikes
                after a long transient state. Therefore, at I_ext = 1/3, we don't see a peak frequency of above 20 due to the long transient state.
                """
                assert peak_freq[j] > 20
            if INTEGRATION_TEST_DEBUG_PLOTS:
                self._timeseries_plot(N1, t_log, h_log, y_log, sym_list, basedir="", fn_snip=" I= " + str(I_ext[j]) + " peaks freq = " + str(peak_freq[j]), title_snip=" I= " + str(I_ext[j]) + " peaks freq= " + str(peak_freq[j]))
        if INTEGRATION_TEST_DEBUG_PLOTS:
            self._FI_curve(I_ext, peak_freq, basedir="", fn_snip="FI curve", title_snip="FI curve")

    def peak_detection(self, y_log, N1, threshold_V_for_peak, time_analysis_start, T):  # function that determines the frequency of peaks in the plot for V vs time
        peaks, _ = find_peaks(np.array(y_log)[N1:, 0], height=threshold_V_for_peak)  # finding peaks above 1.5 microvolts ignoring the first 200 ms
        frequency = int(len(peaks) / ((T - time_analysis_start) * 0.001))  # frequency (in Hz) of the peaks for every value of current
        return frequency

    def _timeseries_plot(self, N1, t_log, h_log, y_log, sym_list, basedir="", fn_snip="", title_snip=""):
        fig, ax = plt.subplots(len(y_log[0]), sharex=True)
        for i, sym in enumerate(sym_list):
            ax[i].plot(np.array(t_log)[N1:], np.array(y_log)[N1:, i], label=str(sym))
        for _ax in ax:
            _ax.legend()
            _ax.grid(True)
        ax[-1].set_xlabel("Time [ms]")
        fig.suptitle("V vs time" + title_snip)
        fn = os.path.join(basedir, "test_fitzhughnagumo" + fn_snip + ".png")
        print("Saving to " + fn)
        plt.savefig(fn, dpi=600)
        plt.close(fig)

    def _FI_curve(self, I_ext, num_peaks, basedir="", fn_snip="", title_snip=""):
        plt.title(title_snip)
        plt.xlabel("External current (arbitrary units)")
        plt.ylabel("Frequency of spikes in Hz")
        plt.plot(I_ext, num_peaks)  # plotting the frequency of peaks vs external current
        fn = os.path.join(basedir, "test_fitzhughnagumo " + fn_snip + ".png")
        print("Saving to " + fn)
        plt.savefig(fn, dpi=600)
        plt.close()


if __name__ == '__main__':
    unittest.main()
