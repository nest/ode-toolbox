# test_cse_mixed_solver_integration.py
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
import odetoolbox
from odetoolbox.mixed_integrator import MixedIntegrator
from tests.test_utils import load_test_json
try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False


class TestCSEMixedSolver:
    """
    Isolated ODE-toolbox validation of CSE for a system containing both an
    analytical solver block and a numerical solver block.
    """

    @pytest.mark.skipif(not PYGSL_AVAILABLE,
                        reason="Need GSL integrator to perform numerical CSE test")
    def test_cse_numerical_integrator_matches_baseline(self):
        """
        Verify that numerical CSE does not change the solution produced
        by MixedIntegrator/GSL.
        """

        indict = load_test_json("cse_mixed.json")

        # baseline _analysis run
        (baseline_solvers, baseline_shape_sys, baseline_shapes) = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_analytic_solver=True,
            disable_singularity_detection=True,
            disable_cse=True,
            log_level=logging.DEBUG)

        # cse _analysis run
        (cse_solvers, cse_shape_sys, cse_shapes) = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_analytic_solver=True,
            disable_singularity_detection=True,
            disable_cse=False,     # default is True, so disable_cse=False has no impact. 
            log_level=logging.DEBUG)

        # assert that _analysis produced solvers
        assert len(baseline_solvers) == 1
        assert len(cse_solvers) == 1
        baseline_solver = baseline_solvers[0]
        cse_solver = cse_solvers[0]

        # assert that solvers correctly identified solver_type
        assert baseline_solver["solver"].startswith("numeric")
        assert cse_solver["solver"].startswith("numeric")

        # Confirm CSE happened
        assert "cse" not in baseline_solver
        assert "cse" in cse_solver
        # numeric so only update expr
        assert ("update_expressions" in cse_solver["cse"])

        # The nonlinear fixture grows quickly, so keep the simulation short.
        simulation_time = 5E-3
        max_step_size = 1E-4

        # construct the mixed integrator baseline run
        baseline_integrator = MixedIntegrator(
            odeiv.step_rk4,
            baseline_shape_sys,
            baseline_shapes,
            analytic_solver_dict=None,
            numeric_solver_dict=baseline_solver,  # baseline solver no cse applied
            # ensure entire dict is the same for params for comparision
            parameters=copy.deepcopy(indict.get("parameters", {})),
            spike_times={},
            random_seed=123,
            max_step_size=max_step_size,
            integration_accuracy_abs=1E-6,
            # accuaracy leads to number of steps simulation steps taken
            integration_accuracy_rel=1E-6,
            sim_time=simulation_time,
            alias_spikes=False)  # tracking of spikes, continuous, or snapping to timesteps?

        # construct the mixed integrator cse run
        cse_integrator = MixedIntegrator(
            odeiv.step_rk4,
            cse_shape_sys,
            cse_shapes,
            analytic_solver_dict=None,
            numeric_solver_dict=cse_solver,  # cse solver provided
            parameters=copy.deepcopy(indict.get("parameters", {})),
            spike_times={},
            random_seed=123,
            max_step_size=max_step_size,
            integration_accuracy_abs=1E-4,
            integration_accuracy_rel=1E-4,
            sim_time=simulation_time,
            alias_spikes=False)

        # Run the GSL simulation on the constructed integators
        baseline_result = (
            baseline_integrator.integrate_ode(
                initial_values={},
                h_min_lower_bound=1E-12,
                raise_errors=True,
                debug=True))

        cse_result = (
            cse_integrator.integrate_ode(
                initial_values={},
                h_min_lower_bound=1E-12,
                raise_errors=True,
                debug=True))

        # unpact tuples from integrate ode
        baseline_t_log = baseline_result[4]  # 1d array of time steps logged
        # 2d array state var values per timestamp
        baseline_y_log = baseline_result[6]
        baseline_symbols = baseline_result[7]  # Symbol mappings
        cse_t_log = cse_result[4]
        cse_y_log = cse_result[6]
        cse_symbols = cse_result[7]

        # Same variables to be integrated, ensures that cse has not broken the code 
        assert ([str(symbol) for symbol in baseline_symbols] == [str(symbol) for symbol in cse_symbols])

        # execution check ensure that both simulations reached the requested time
        np.testing.assert_allclose(baseline_t_log[-1], simulation_time)
        np.testing.assert_allclose(cse_t_log[-1], simulation_time)

        # use rtol and atol for numerical thresholds to ensure cse has not
        # broken the mathematics of the ode
        # compare state variables across all timepoints
        np.testing.assert_allclose(
            cse_y_log, baseline_y_log, rtol=1E-6, atol=1E-8)

        print("Numerical baseline final:", baseline_y_log[-1])
        print("Numerical CSE final:", cse_y_log[-1])
