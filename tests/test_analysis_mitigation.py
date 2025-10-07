# tests/test_integration_branch_flow_numeric.py
from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_allclose

from odetoolbox.singularity_analysis_mitigation import (
    build_transition_branches_numeric,
    BranchPack,
)


# Utilities (numeric version)

def jordan_expected_numeric(alpha: float, h: float) -> np.ndarray:
    """
    Expected Jordan-form propagator for the degenerate case a = b.

    System:
        A = [[ alpha, 0, 0],
             [   1  , alpha, 0],
             [   0  ,   1  , alpha]]

    Theoretical propagator:
        P = exp(-h*A) = exp(-h*alpha) * [[1, 0,   0],
                                         [h, 1,   0],
                                         [h*h/2, h, 1]]
    """
    E = np.exp(-h * alpha)
    return E * np.array([[1.0,      0.0, 0.0],
                         [h,        1.0, 0.0],
                         [0.5*h*h,  h,   1.0]], dtype=float)


TOL = dict(rtol=1e-10, atol=1e-12)



# 1) Direct verification: when a = b, the branch propagator

def test_numeric_branch_yields_jordan():
    # Equivalent system to the original symbolic test, but defined as a numeric function A_fn.
    # Original:  x' = -a x,  y' = x - a y,  z' = y - b z
    # Numeric version: A = [[ a, 0, 0],
    #                       [-1, a, 0],
    #                       [ 0, -1, b]]
    # This uses P = exp(-h*A), equivalent in meaning to the symbolic exp(-hA) form.
    def A_fn(p):
        a, b = p["a"], p["b"]
        return np.array([[a,   0.0, 0.0],
                         [-1.0, a,   0.0],
                         [0.0, -1.0, b  ]], dtype=float)

    h = 0.2
    base = {"a": 1.3, "b": 2.1}

    pack: BranchPack = build_transition_branches_numeric(
        A_fn=A_fn,
        h=h,
        base_params=base,
        param_names=["a", "b"],
        conditions=[("a", "b")],   # Only test the condition a = b
        tie_mode="left",           # Overwrite b with a’s value
    )

    # Extract the a=b branch
    assert len(pack.branches) == 1
    br = pack.branches[0]
    assert tuple(br.condition) == ("a", "b")

    # Expected Jordan-form propagator for a=b=base["a"]
    alpha = base["a"]
    P_expected = jordan_expected_numeric(alpha, h)

    # Numerical equivalence check
    assert_allclose(br.P, P_expected, **TOL)



# 2) Auto-generated “pairwise equality” condition test

def test_numeric_branch_auto_conditions():
    def A_fn(p):
        # Simple 3×3 upper-triangular chain structure
        t1, t2, t3 = p["t1"], p["t2"], p["t3"]
        return np.array([[t1,   0.0, 0.0],
                         [-1.0, t2,  0.0],
                         [0.0, -1.0, t3]], dtype=float)

    h = 0.05
    base = {"t1": 1.0, "t2": 1.0, "t3": 2.0}  

    pack = build_transition_branches_numeric(
        A_fn=A_fn,
        h=h,
        base_params=base,
        param_names=["t1", "t2", "t3"],
        conditions=None,           
        tie_mode="left",
    )

    # The automatically generated condition list should include ('t1', 't2')
    conds = [tuple(c.condition) for c in pack.branches]
    assert ("t1", "t2") in conds

    # Find the ('t1','t2') branch and verify that its top-left 2×2 block
    # matches the analytical 2×2 Jordan block for parameter alpha = t1.
    br = next(c for c in pack.branches if tuple(c.condition) == ("t1", "t2"))
    alpha = base["t1"]
    P_expected_2x2 = np.exp(-h*alpha) * np.array([[1.0, 0.0],
                                                  [h,   1.0]])
    assert_allclose(br.P[:2, :2], P_expected_2x2, **TOL)


# 3) Basic robustness test: shape and near-identity behavior

def test_numeric_default_shape_and_sanity():
    def A_fn(p):
        return np.array([[p["a"], 1.0],
                         [0.0,     p["b"]]], dtype=float)

    h = 0.1
    base = {"a": 1.0, "b": 1.0}

    pack = build_transition_branches_numeric(
        A_fn=A_fn, h=h, base_params=base,
        param_names=["a", "b"], conditions=[("a","b")]
    )

    P0 = pack.default.P
    assert P0.shape == (2, 2)

    # For small h, exp(-hA) ≈ I - hA; this is a mild sanity check (not a proof)
    I = np.eye(2)
    approx = I - h * A_fn(base)
    # Only check magnitude consistency, not exact equality
    assert np.linalg.norm(P0 - approx) < 1.0



