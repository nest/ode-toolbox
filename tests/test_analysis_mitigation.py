# tests/test_integration_branch_flow_numeric.py
from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_allclose

from odetoolbox.singularity_analysis_mitigation import (
    build_transition_cases,
    CaseSet,
)

# ----------------- helpers -----------------

def _call_build_transition_cases(**kwargs):
    """
    Compatibility shim:
    - If build_transition_cases returns a CaseSet, wrap it as (CaseSet, None)
    - If it already returns (CaseSet, assignments), pass through.
    """
    out = build_transition_cases(**kwargs)
    if isinstance(out, CaseSet):
        return out, None
    # tuple or other iterable with CaseSet first
    try:
        cs, assignments = out
        if isinstance(cs, CaseSet):
            return cs, assignments
    except Exception:
        pass
    raise TypeError("build_transition_cases must return CaseSet or (CaseSet, assignments)")

def _get_condition_tuple(case) -> tuple[str, str] | None:
    """
    Return ('a','b') if exactly one equality is active; otherwise None.
    Works with either:
      - case.condition (if provided by implementation), or
      - case.active_equalities (frozenset of pairs)
    """
    if hasattr(case, "condition") and case.condition is not None:
        return tuple(case.condition)  
    if hasattr(case, "active_equalities"):
        eqs = getattr(case, "active_equalities")
        if isinstance(eqs, (set, frozenset)) and len(eqs) == 1:
            return tuple(next(iter(eqs)))
    return None

# ----------------- utilities -----------------

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

# ----------------- tests -----------------

def test_numeric_case_yields_jordan():
    """
    Check that when a=b, the propagated matrix matches the analytical Jordan form.
    """
    def A_fn(p):
        a, b = p["a"], p["b"]
        return np.array([[a,   0.0, 0.0],
                         [-1.0, a,   0.0],
                         [0.0, -1.0, b  ]], dtype=float)

    h = 0.2
    base = {"a": 1.3, "b": 2.1}

    case_set, _ = _call_build_transition_cases(
        A_fn=A_fn,
        h=h,
        base_params=base,
        param_names=["a", "b"],
        conditions=[("a", "b")],   
        tie_mode="left",           
        expand_truth_table=False,  
    )

    # Extract the a=b case
    assert len(case_set.cases) == 1
    case = case_set.cases[0]
    assert _get_condition_tuple(case) == ("a", "b")

    # Expected Jordan-form propagator for a=b=base["a"]
    alpha = base["a"]
    P_expected = jordan_expected_numeric(alpha, h)

    # Numerical equivalence check
    assert_allclose(case.P, P_expected, **TOL)


def test_numeric_case_auto_conditions():
    """
    Verify that the automatically generated pairwise equality conditions
    include ('t1', 't2') and that its sub-block matches the 2×2 Jordan form.
    """
    def A_fn(p):
        # Simple 3×3 upper-triangular chain structure
        t1, t2, t3 = p["t1"], p["t2"], p["t3"]
        return np.array([[t1,   0.0, 0.0],
                         [-1.0, t2,  0.0],
                         [0.0, -1.0, t3]], dtype=float)

    h = 0.05
    base = {"t1": 1.0, "t2": 1.0, "t3": 2.0}

    case_set, _ = _call_build_transition_cases(
        A_fn=A_fn,
        h=h,
        base_params=base,
        param_names=["t1", "t2", "t3"],
        conditions=None,
        tie_mode="left",
        expand_truth_table=False,
    )

    # The automatically generated condition list should include ('t1', 't2')
    conds = [_get_condition_tuple(c) for c in case_set.cases]
    assert ("t1", "t2") in conds

    # Find the ('t1','t2') case and verify that its top-left 2×2 block
    # matches the analytical 2×2 Jordan block for parameter alpha = t1.
    c = next(c for c in case_set.cases if _get_condition_tuple(c) == ("t1", "t2"))
    alpha = base["t1"]
    P_expected_2x2 = np.exp(-h*alpha) * np.array([[1.0, 0.0],
                                                  [h,   1.0]])
    assert_allclose(c.P[:2, :2], P_expected_2x2, **TOL)


def test_numeric_default_shape_and_sanity():
    """
    For small h, exp(-hA) ≈ I - hA. This serves as a numerical sanity check.
    """
    def A_fn(p):
        return np.array([[p["a"], 1.0],
                         [0.0,     p["b"]]], dtype=float)

    h = 0.1
    base = {"a": 1.0, "b": 1.0}

    case_set, _ = _call_build_transition_cases(
        A_fn=A_fn,
        h=h,
        base_params=base,
        param_names=["a", "b"],
        conditions=[("a", "b")],
        expand_truth_table=False,
    )

    P0 = case_set.default.P
    assert P0.shape == (2, 2)

    # For small h, exp(-hA) ≈ I - hA; mild consistency check
    I = np.eye(2)
    approx = I - h * A_fn(base)
    assert np.linalg.norm(P0 - approx) < 1.0
