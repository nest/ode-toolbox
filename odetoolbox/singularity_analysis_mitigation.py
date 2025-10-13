# odetoolbox/singularity_analysis_mitigation.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Tuple, Optional, Set, FrozenSet, Any
from itertools import product
import json

import numpy as np
from scipy.linalg import expm


# Type definitions
Params      = Dict[str, float]                 
Cond        = Tuple[str, str]                   
AFunction   = Callable[[Params], np.ndarray]   
SymRenderer = Callable[[FrozenSet[Cond], str], Any] 


# Config (lightweight; no external deps)
class Config:
    """
    Minimal configuration shim to avoid if-else on timestep symbols.
    You may inject your own symbol string via build_* API. This class
    exists only to provide a default.
    """
    @staticmethod
    def default_timestep_symbol() -> str:
        return "h"


# Data structures
@dataclass(frozen=True)
class CaseResult:
    """
    One case under a set of active equality constraints.
    - active_equalities: frozenset of normalized pairs (min(a,b), max(a,b))
    - P: optional numeric preview of the propagator (expm(-h*A))
    - symbolic: optional symbolic/IR object produced by a renderer (no sympy required)
    """
    active_equalities: FrozenSet[Cond]
    P: Optional[np.ndarray] = None
    symbolic: Optional[Any] = None

@dataclass
class CaseSet:
    """
    Container of the default (no constraints) and all expanded cases.
    """
    timestep_symbol: str
    default: CaseResult
    cases: List[CaseResult]
    expanded: bool
    truncated: bool
    max_cases: Optional[int]


# Utilities
def _ensure_square(A: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")

def _matrix_exp(A: np.ndarray, h: float) -> np.ndarray:
    _ensure_square(A)
    return expm(-h * A)

def _normalize_conds(conditions: Iterable[Cond]) -> List[Cond]:
    """
    Normalize conditions into sorted, deduplicated pairs so that ("b","a") -> ("a","b").
    """
    norm = sorted({tuple(sorted((a, b))) for (a, b) in conditions})
    # ensure no self-equalities like ("a","a")
    for a, b in norm:
        if a == b:
            raise ValueError(f"Invalid equality condition ({a},{b}): identical names are not allowed.")
    return norm

def _expand_truth_table(
    conds: List[Cond],
    include_empty: bool = False,
    max_cases: Optional[int] = None
) -> List[FrozenSet[Cond]]:
    """
    Return all truth assignments as the set of 'true' equalities.
    If include_empty is False, the empty set is skipped.
    """
    if not conds:
        return [frozenset()] if include_empty else []

    true_sets: List[FrozenSet[Cond]] = []
    for bits in product([False, True], repeat=len(conds)):
        active = frozenset(conds[i] for i, flag in enumerate(bits) if flag)
        if not include_empty and len(active) == 0:
            continue
        true_sets.append(active)
        if max_cases is not None and len(true_sets) >= max_cases:
            break
    return true_sets

def _apply_numeric_ties(base_params: Params, true_eqs: FrozenSet[Cond], tie_mode: str = "left") -> Params:
    """
    Numeric-layer tie for a set of equalities:
      - 'left'  : set right := left (b := a)
      - 'right' : set left  := right (a := b)
    """
    out = dict(base_params)
    for (a, b) in sorted(true_eqs):
        if a not in out or b not in out:
            raise KeyError(f"Missing parameter for tie: {a} or {b}")
        if tie_mode == "left":
            out[b] = out[a]
        elif tie_mode == "right":
            out[a] = out[b]
        else:
            raise ValueError("tie_mode must be 'left' or 'right'")
    return out

def _infer_all_pairwise(param_names: Iterable[str]) -> List[Cond]:
    names = list(param_names)
    conds: List[Cond] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a != b:
                conds.append(tuple(sorted((a, b))))
    return sorted(set(conds))


# Main API
def build_transition_cases(
    A_fn: AFunction,
    h: float,
    base_params: Params,
    *,
    # condition sources
    conditions: Optional[Iterable[Cond]] = None,
    param_names: Optional[Iterable[str]] = None,
    # expansion behaviour
    expand_truth_table: bool = True,
    max_cases: Optional[int] = 4096,
    # numeric preview behaviour
    numeric_preview: bool = True,
    tie_mode: str = "left",
    # symbolic behaviour 
    timestep_symbol: Optional[str] = None,
    symbolic_renderer: Optional[SymRenderer] = None,
) -> CaseSet:
    """
    Construct the default case plus all expanded cases from
    the truth-table of provided conditions.

    - No sympy reliance. If you need a symbolic object, supply `symbolic_renderer`.
    - Numeric preview uses expm(-h*A(params_tied)) for each case.
    - Conditions may be explicitly provided. If omitted, and `param_names` is given,
      we infer all pairwise equalities.

    Returns:
        CaseSet with:
          - default: active_equalities = empty set
          - cases: list of CaseResult with active_equalities carrying the constraints
          - expanded/truncated/max_cases metadata
    """
    step_sym = timestep_symbol or Config.default_timestep_symbol()

    #default case
    A0 = np.asarray(A_fn(dict(base_params)), dtype=float)
    P0 = _matrix_exp(A0, h) if numeric_preview else None
    default = CaseResult(active_equalities=frozenset(), P=P0,
                         symbolic=(symbolic_renderer(frozenset(), step_sym) if symbolic_renderer else None))

    #canonicalize conditions
    if conditions is None:
        if not param_names:
            # no conditions at all
            return CaseSet(
                timestep_symbol=step_sym,
                default=default,
                cases=[],
                expanded=False,
                truncated=False,
                max_cases=None,
            )
        else:
            conditions = _infer_all_pairwise(param_names)

    conds = _normalize_conds(conditions)

    # expansion list
    if not expand_truth_table:
        # one-at-a-time activation
        actives = [frozenset([c]) for c in conds]
        truncated = False
    else:
        actives = _expand_truth_table(conds, include_empty=False, max_cases=max_cases)
        truncated = (max_cases is not None and len(actives) >= max_cases)

    #build cases
    cases: List[CaseResult] = []
    for active_eqs in actives:
        params_ = _apply_numeric_ties(base_params, active_eqs, tie_mode=tie_mode) if numeric_preview else base_params
        A_ = np.asarray(A_fn(params_), dtype=float) if numeric_preview else None
        P_ = _matrix_exp(A_, h) if (numeric_preview and A_ is not None) else None
        sym_obj = symbolic_renderer(active_eqs, step_sym) if symbolic_renderer else None
        cases.append(CaseResult(active_equalities=active_eqs, P=P_, symbolic=sym_obj))

    return CaseSet(
        timestep_symbol=step_sym,
        default=default,
        cases=cases,
        expanded=True,
        truncated=truncated,
        max_cases=max_cases if expand_truth_table else None,
    )

# JSON encoding helpers
def _ndarray_to_list(arr: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if arr is None:
        return None
    return arr.tolist()

def case_set_to_json(case_set: CaseSet) -> str:
    payload = {
        "timestep_symbol": case_set.timestep_symbol,
        "expanded": case_set.expanded,
        "truncated": case_set.truncated,
        "max_cases": case_set.max_cases,
        "default": {
            "active_equalities": sorted(list(case_set.default.active_equalities)),
            "P": _ndarray_to_list(case_set.default.P),
            "symbolic": case_set.default.symbolic if _is_jsonable(case_set.default.symbolic) else None,
        },
        "cases": [
            {
                "active_equalities": sorted(list(cr.active_equalities)),
                "P": _ndarray_to_list(cr.P),
                "symbolic": cr.symbolic if _is_jsonable(cr.symbolic) else None,
            }
            for cr in case_set.cases
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

def _is_jsonable(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except Exception:
        return False



