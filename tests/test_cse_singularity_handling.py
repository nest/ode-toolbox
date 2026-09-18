# test_cse_singularity_handling.py
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
# This file is part of the NEST ODE toolbox.
#

import copy
import logging
import numpy as np
import pytest
import json
import odetoolbox
from odetoolbox.analytic_integrator import (AnalyticIntegrator)
from tests.test_utils import load_test_json
try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False


class TestCSESingularityHandling:
    """
    Isolated ODE-toolbox validation of CSE for a system containing both an
    analytical solver block and a numerical solver block.
    """

    @pytest.mark.skipif(not PYGSL_AVAILABLE,
                        reason="Need GSL integrator to perform numerical CSE test")
    def test_cse_singularity_handling(self):
        """
        Verify that a conditional cse json vs a baseline conditional handling does not change the solution produced
        """

        # load in json that will produce a conditional argument
        indict = load_test_json("conditional.json")

        # baseline _analysis run
        (baseline_solvers, baseline_shape_sys, baseline_shapes) = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_cse=True,     # disable cse, so set this flag to true. 
            log_level=logging.DEBUG)

        # cse _analysis run
        (cse_solvers, cse_shape_sys, cse_shapes) = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_cse=False,
            log_level=logging.DEBUG)

        # ensure solver was identified as analytical
        baseline_solver = next(
            s for s in baseline_solvers if s["solver"] == "analytical")
        cse_solver = next(
            s for s in cse_solvers if s["solver"] == "analytical")

        # confirm both solvers have singularity conditions
        assert "conditions" in baseline_solver
        assert "conditions" in cse_solver

        for cond_key, base_branch in baseline_solver["conditions"].items(
        ):  # baseline branches should be raw (no CSE temporaries),
            assert "cse" not in base_branch, f"unexpected CSE temporaries in baseline branch {cond_key}"

        for cond_key, cse_branch in cse_solver["conditions"].items(
        ):  # cse branches should each carry their own "cse" sub-dict
            assert "cse" in cse_branch, f"expected CSE temporaries in cse branch {cond_key}"

        # keep the simulation short for testing
        simulation_time = 5E-3
        max_step_size = 1E-4

        # parameters to produce a singularity, division by 0, where tm = tsyn (2.0)
        params_singular = {"tau_syn": "2.0", "tau_m": "2.0", "C_m": "250.0"}
        # parameters to produce a default solver, where tm does not equal tsyn
        params_default = {"tau_syn": "2.0", "tau_m": "5.0", "C_m": "250.0"}

        baseline_solver.setdefault("parameters", {})
        cse_solver.setdefault("parameters", {})

        # creating 51 steps for simulation time
        time_grid = np.linspace(0.0, simulation_time, 51)

        # run two seperate common, singularity solver simulations
        for label, params in [
                ("singular", params_singular), ("default", params_default)]:
            # update parameters based on the current simulation
            baseline_solver["parameters"].update(params)
            cse_solver["parameters"].update(params)

            # Run through the existing analytical integrator pipeline passing
            # baseline and cse solvers
            baseline_integrator = AnalyticIntegrator(baseline_solver)
            cse_integrator = AnalyticIntegrator(cse_solver)

            for t in time_grid:    # structural check checking param tracking across gird
                baseline_state = baseline_integrator.get_value(t)
                cse_state = cse_integrator.get_value(t)
                assert baseline_state.keys() == cse_state.keys()

                for symbol in baseline_state:    # ensure that they are numerically exact across all states 
                    np.testing.assert_allclose(
                        cse_state[symbol],
                        baseline_state[symbol],
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=f"CSE diverged from baseline on '{label}' branch at t={t}")
