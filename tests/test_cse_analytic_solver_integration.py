#
# test_cse_analytic_solver_integration.py
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
import sympy
import odetoolbox
from odetoolbox.analytic_integrator import (AnalyticIntegrator)
from tests.test_utils import load_test_json
import json 


try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True

except ImportError:
    PYGSL_AVAILABLE = False


class TestCSENumericalSolver:
    """
    Isolated ODE-toolbox validation of CSE applied to a numerical solver.
    """

    def test_cse_analytic_integrator_matches_baseline(self):
        """
        Verify that analytical CSE does not change the trajectory produced
        by ODE-toolbox's AnalyticIntegrator.
        """

        indict = load_test_json("cse_analytical.json")

        # run baseline _analysis (no cse applied)
        baseline_solvers, _, _ = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_singularity_detection=True,
            disable_cse=True,    # disable cse so set this to true 
            # specified as false (default is already false)
            log_level=logging.DEBUG)

        # run baseline _analysis (no cse applied)
        cse_solvers, _, _ = odetoolbox._analysis(
            copy.deepcopy(indict),
            disable_stiffness_check=True,
            disable_singularity_detection=True,
            disable_cse=False,     # specified as true for cse
            log_level=logging.DEBUG)

        # verify _analysis produced solvers
        assert len(baseline_solvers) == 1
        assert len(cse_solvers) == 1
        baseline_solver = baseline_solvers[0]
        cse_solver = cse_solvers[0]

        # verify the solver_type was correctly identified as analytical
        assert baseline_solver["solver"] == "analytical"
        assert cse_solver["solver"] == "analytical"

        # Prove that cse occured
        assert "cse" not in baseline_solver
        assert "cse" in cse_solver
        assert "propagators" in cse_solver["cse"]

        # Make parameter values explicitly available to AnalyticIntegrator.
        baseline_solver.setdefault("parameters", {})
        cse_solver.setdefault("parameters", {})
        # pass parameters into the baseline solvers as they contain expr
        baseline_solver["parameters"].update(indict.get("parameters", {}))
        cse_solver["parameters"].update(indict.get("parameters", {}))

        # Run through the existing analytical integrator pipeline passing
        # baseline and cse solvers
        baseline_integrator = AnalyticIntegrator(baseline_solver)
        cse_integrator = AnalyticIntegrator(cse_solver)

        # generate 101 timepoints between 0-20
        time_points = np.linspace(0.0, 20.0, 101)

        for t in time_points:

            # for time t, check that both states have outputted the exact
            # variables/symbols from the propagator
            baseline_state = (baseline_integrator.get_value(t))
            cse_state = (cse_integrator.get_value(t))
            assert (baseline_state.keys() == cse_state.keys())

            # iterate through every single symbol at every individual time
            # point to compare their values.
            for symbol in baseline_state:
                np.testing.assert_allclose(
                    cse_state[symbol],
                    baseline_state[symbol],
                    rtol=1E-10,
                    atol=1E-12)
                # asset a higher tolerance to the analytical algebraic
                # substituion with no approximation

        # print final vectors evaluated at the last time point
        print("Analytical baseline final:", baseline_state)
        print("Analytical CSE final:", cse_state)
