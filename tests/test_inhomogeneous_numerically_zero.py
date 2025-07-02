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

import sympy
import sympy.parsing.sympy_parser
import pytest

from tests.test_utils import _open_json

from .context import odetoolbox
from odetoolbox.analytic_integrator import AnalyticIntegrator
from odetoolbox.shapes import Shape
import numpy as np
try:
    import pygsl
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False


import matplotlib.pyplot as plt

class TestInhomogeneousNumericallyZero:
    @pytest.mark.parametrize("late_ltd_check", [0., 1.])
    @pytest.mark.parametrize("late_ltp_check", [0., 1.])
    def test_inhomogeneous_numerically_zero(self, late_ltd_check, late_ltp_check):
        """This tests especially the case where there is an inhomogeneous component given in the system to be integrated, but during numerical integration the inhomogeneous part is set to 0.

        Thus, especially the case late_ltd_check = 0, late_ltp_check = 0 is the critical one.
        """
        indict = {
  "dynamics": [
    {
      "expression": "z' = ((p * (1 - z) * late_ltp_check) - (p * (z + 0.5) * late_ltd_check)) / tau_z",
      "initial_value" : "1"
    }
  ],
}
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

        analytic_integrator = AnalyticIntegrator(solver_dict)
        analytic_integrator.set_initial_values({"z": 0.})
        analytic_integrator.reset()

        dt = .1
        T = 100.

        actual = []
        correct = []
        cur_z = 0.
        timevec = np.arange(0., T, dt)
        # XXX: TODO: do the integration using scipy runge-kutta
        for step, t in enumerate(timevec):
            state_ = analytic_integrator.get_value(t)["z"]
            actual.append(state_)

            correct.append(cur_z)
            cur_z += dt * (((solver_dict["parameters"]["p"] * (1 - cur_z) * solver_dict["parameters"]["late_ltp_check"]) - (solver_dict["parameters"]["p"] * (cur_z + 0.5) * solver_dict["parameters"]["late_ltd_check"])) / solver_dict["parameters"]["tau_z"])

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

        np.testing.assert_allclose(correct, actual)

        import pdb;pdb.set_trace()
        """assert sympy.parsing.sympy_parser.parse_expr(solver_dict["update_expressions"]["y"], global_dict=Shape._sympy_globals).expand().simplify() \
            == sympy.parsing.sympy_parser.parse_expr("rho*x - x*z - y", global_dict=Shape._sympy_globals).expand().simplify()
        assert sympy.parsing.sympy_parser.parse_expr(solver_dict["update_expressions"]["z"], global_dict=Shape._sympy_globals).expand().simplify() \
            == sympy.parsing.sympy_parser.parse_expr("-beta*z + x*y", global_dict=Shape._sympy_globals).expand().simplify()"""
