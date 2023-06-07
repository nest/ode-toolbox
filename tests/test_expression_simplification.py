#
# test_expression_simplification.py
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

import numpy as np
import pytest
from tests.test_mixed_integrator_numeric import _run_simulation
from tests.test_utils import _open_json

import odetoolbox

try:
    import pygsl.odeiv as odeiv
    PYGSL_AVAILABLE = True
except ImportError:
    PYGSL_AVAILABLE = False


@pytest.mark.skipif(not PYGSL_AVAILABLE, reason="Need GSL integrator to perform test")
def test_expression_simplification():
    """
    Test expression simplification features: run ODE-toolbox for various combinations of ``preserve_expression`` and ``simplify_expression``, and check that numerical simulation results are the same, even though the returned update expressions could be different.

    Note that this test uses all-numeric (no analytic part) integration.
    """
    opts = [(True, "sympy.simplify(expr)"),
            (False, "sympy.simplify(expr)"),
            (True, "sympy.logcombine(sympy.powsimp(sympy.expand(expr)))")]
    ts = {}
    for preserve_expressions, simplify_expression in opts:
        print("Running test with preserve_expressions = " + str(preserve_expressions) + ", simplify_expression = " + str(simplify_expression))

        indict = _open_json("eiaf_cond_alpha.json")
        if "options" not in indict.keys():
            indict["options"] = {}
        indict["options"]["simplify_expression"] = simplify_expression

        _, _, _, _, t_log, _, y_log, _, analysis_json = _run_simulation(indict, alias_spikes=False, integrator=odeiv.step_rk4, preserve_expressions=preserve_expressions)
        ts[(preserve_expressions, simplify_expression)] = y_log
        print("\t-> expr = " + analysis_json[0]["update_expressions"]["V_m"])

    x_ref = list(ts.values())[0]
    for x in ts.values():
        np.testing.assert_allclose(x, x_ref)


def test_expression_simplification_analytic():
    """
    Test expression simplification: test that ``preserve_expression`` is ignored for equations that are solved analytically
    """

    indict = {"dynamics": [{"expression": "x' = -x / 42",
                            "initial_value": "42"}]}

    analysis_json = odetoolbox.analysis(indict, preserve_expressions=True, log_level="DEBUG")

    assert "__P__x__x" in analysis_json[0]["update_expressions"]["x"]
