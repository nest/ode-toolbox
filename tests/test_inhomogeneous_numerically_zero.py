#
# test_inhomogeneous_numerically_zero.py
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

import pytest

import numpy as np
import scipy.integrate

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    DEBUG_PLOTS = True
except ImportError:
    DEBUG_PLOTS = False

from .context import odetoolbox
from odetoolbox.analytic_integrator import AnalyticIntegrator


class TestInhomogeneousNumericallyZero:
    def _test_inhomogeneous_numerically_zero(self, late_ltd_check, late_ltp_check):
        """This tests the case where there is an inhomogeneous component given in the system to be integrated, but during numerical integration the inhomogeneous part is set to 0.

        ODE-toolbox emits a warning in this case. If the propagators are used in this case, integration will fail (yield NaNs due to division by zero). Thus, two of the tests are marked as xfail.
        """
        indict = {"dynamics": [{"expression": "z' = ((p * (1 - z) * late_ltp_check) - (p * (z + 0.5) * late_ltd_check)) / tau_z",
                                "initial_value": "1"}]}
        solver_dict = odetoolbox.analysis(indict, disable_stiffness_check=True, log_level="DEBUG")
        assert len(solver_dict) == 1
        solver_dict = solver_dict[0]
        assert solver_dict["solver"].startswith("analytic")
        print(solver_dict)

        solver_dict["parameters"] = {}
        solver_dict["parameters"]["p"] = .5
        solver_dict["parameters"]["tau_z"] = 20.
        solver_dict["parameters"]["late_ltd_check"] = late_ltd_check
        solver_dict["parameters"]["late_ltp_check"] = late_ltp_check

        dt = .1
        T = 100.
        timevec = np.arange(0., T, dt)

        #
        #    integration using the ODE-toolbox analytic integrator
        #

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"z": 0.})
        analytic_integrator.reset()
        actual = [analytic_integrator.get_value(t)["z"] for t in timevec]


        #
        #    integration using scipy.integrate.odeint
        #


        def ode_model(z, t, p, late_ltp_check, late_ltd_check, tau_z):
            """
            Defines the differential equation for z.
            dz/dt = f(z, t)
            """
            dzdt = (((p * (1.0 - z) * late_ltp_check) - (p * (z + 0.5) * late_ltd_check))) / tau_z
            return dzdt

        z0 = 0.0      # set the initial condition
        params = solver_dict["parameters"]
        ode_args = (
            params["p"],
            params["late_ltp_check"],
            params["late_ltd_check"],
            params["tau_z"]
        )

        solution = scipy.integrate.odeint(ode_model, z0, timevec, args=ode_args, rtol=1E-12, atol=1E-12)
        correct = solution.flatten().tolist()


        #
        #   plot
        #

        if DEBUG_PLOTS:
            fig, ax = plt.subplots(nrows=3)
            ax[0].plot(timevec, correct, label="reference")
            ax[1].plot(timevec, actual, label="actual")
            ax[2].semilogy(timevec, np.abs(np.array(correct) - np.array(actual)))
            ax[-1].set_xlabel("Time")
            for _ax in ax:
                _ax.set_xlim(0, T)
                _ax.legend()
                _ax.grid()
                if not _ax == ax[-1]:
                    _ax.set_xticklabels([])

            fig.savefig("/tmp/test_propagators_[late_ltd_check=" + str(late_ltd_check) + "]_[late_ltp_check=" + str(late_ltp_check) + "].png")


        #
        #   test
        #

        np.testing.assert_allclose(correct, actual)

    @pytest.mark.xfail(strict=True, raises=AssertionError)
    def test_inhomogeneous_numerically_zero(self):
        self._test_inhomogeneous_numerically_zero(late_ltd_check=1., late_ltp_check=-1.)

    @pytest.mark.xfail(strict=True, raises=AssertionError)
    def test_inhomogeneous_numerically_zero_alt(self):
        self._test_inhomogeneous_numerically_zero(late_ltd_check=0., late_ltp_check=0.)

    def test_inhomogeneous_numerically_zero_alt_nonzero(self):
        self._test_inhomogeneous_numerically_zero(late_ltd_check=3.14159, late_ltp_check=1.618)
