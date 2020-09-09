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
from scipy.signal import find_peaks
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve
from sympy import Symbol


try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    INTEGRATION_TEST_DEBUG_PLOTS = True
except:
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
    Implementing the fitzhughNagumo model starting from equilibrium values, and performing a test that if the external current crosses a certain threshold value, regular spikes are obtained.
    This function tests if the number of spikes cross 20 in that case. 
    Additionally, plots of V and W vs time are obtained for different values of current, and a FI curve is also plotted. 
    """
    
    def initial__values(self, curr):
       I_ext = Symbol("I_ext")
       V = Symbol("V") 
       expr = solve((sympy.parsing.sympy_parser.parse_expr("8*V**3 + 6*V + 21 - 24*I_ext")), V) # expr gives a list of three roots for V: first two are complex, third one is real
       final_val_V = (expr[2].subs(I_ext,curr)).evalf()
       final_val_W = ((10*final_val_V) + 7)/8
       return float(final_val_V), float(final_val_W)

    @pytest.mark.skipif(not PYGSL_AVAILABLE, reason="Need GSL integrator to perform test")
    def test_fitzhugh_nagumo(self):
        debug = True

        h = 1    # [ms] #time steps
        T = 1000  # [ms] #total simulation time 
        n = 25 #total number of current values between 0 and 1
        I_ext = np.linspace(0,1,n) #external current
        time_analysis_start = 200 #starting our ananlysis after 200 ms
        N1 = (int)((time_analysis_start)/h) #index of the starting time
        num_peaks = np.zeros(n)
        indict = open_json("fitzhughnagumo.json")
        analysis_json, shape_sys, shapes = odetoolbox._analysis(indict, disable_stiffness_check=True, disable_analytic_solver=True)
        print("Got analysis result from ode-toolbox: ")
        print(json.dumps(analysis_json, indent=2))
        assert len(analysis_json) == 1
        assert analysis_json[0]["solver"].startswith("numeric")
        alias_spikes = True 
        integrator = odeiv.step_rk4
        
        for j in range(n):
            #loop over current values
            initial_values = { "V" : (self.initial__values(I_ext[j])[0] + 0.001), "W": self.initial__values(I_ext[j])[1]}
            initial_values = { sympy.Symbol(k) : v for k, v in initial_values.items() }
            mixed_integrator = MixedIntegrator(
             integrator,
             shape_sys,
             shapes,
             analytic_solver_dict=None,
             parameters={"I_ext":str(I_ext[j])},
             random_seed=123,
             max_step_size=h,
             integration_accuracy_abs=1E-5,
             integration_accuracy_rel=1E-5,
             sim_time=T,
             alias_spikes=alias_spikes)
            h_min, h_avg, runtime, upper_bound_crossed, t_log, h_log, y_log, sym_list = mixed_integrator.integrate_ode(
             initial_values=initial_values,
             h_min_lower_bound=1E-12, raise_errors=True, debug=True) # debug needs to be True here to obtain the right return values
            peaks, _ = find_peaks(np.array(y_log)[N1:,0], height = 1.5 ) #finding peaks above 1.5 microvolts ignoring the first 200 ms
            num_peaks[j] = (int)(len(peaks)/((T-200)*0.001)) #frequency (in Hz) of the peaks for every value of current
            if(I_ext[j] >(1/3)):
                assert(num_peaks[j]>20)
            if INTEGRATION_TEST_DEBUG_PLOTS:
                self._timeseries_plot(N1,t_log, h_log, y_log, sym_list, basedir="", fn_snip = " I= " + str(I_ext[j]) + " peaks= " + str(num_peaks[j]), title_snip= " I= " + str(I_ext[j]) + " peaks= " + str(num_peaks[j]))
        if INTEGRATION_TEST_DEBUG_PLOTS:
            self._FI_curve(I_ext,num_peaks,basedir="",fn_snip = "FI curve", title_snip = "FI curve")

    def _timeseries_plot(self,N1, t_log, h_log, y_log, sym_list, basedir="", fn_snip="", title_snip=""):
        fig, ax = plt.subplots(len(y_log[0]), sharex=True)
        for i, sym in enumerate(sym_list):
            ax[i].plot(np.array(t_log)[N1:], np.array(y_log)[N1:, i], label=str(sym))

        for _ax in ax:
            _ax.legend()
            _ax.grid(True)

        ax[-1].set_xlabel("Time [ms]")
        fig.suptitle("V vs time" + title_snip)

        fn = os.path.join(basedir, "test_fitzhughnagumo"  + fn_snip + ".png")
        print("Saving to " + fn)
        plt.savefig(fn, dpi=600)
        plt.close(fig)
        
    def _FI_curve(self,I_ext,num_peaks,basedir="",fn_snip="",title_snip=""):
        plt.title(title_snip)
        plt.xlabel("External current (arbitrary units)")
        plt.ylabel("Frequency of spikes in Hz")
        plt.plot(I_ext, num_peaks) #plotting the frequency of peaks vs external current
        fn = os.path.join(basedir, "test_fitzhughnagumo " + fn_snip + ".png")
        print("Saving to " + fn)
        plt.savefig(fn,dpi=600)
        plt.close()
       
        


if __name__ == '__main__':
    unittest.main()
