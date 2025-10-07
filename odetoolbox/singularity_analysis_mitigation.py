# odetoolbox/singularity_analysis_mitigation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Optional
import numpy as np
from scipy.linalg import expm

# ---- Type definitions ----
Params = Dict[str, float]                 
Cond = Tuple[str, str]                     
AFunction = Callable[[Params], np.ndarray] 


# ---- Data structures ----
@dataclass
class BranchResult:
    """Represents the result of one branch: a condition and its corresponding propagator matrix."""
    condition: Optional[Cond]              
    P: np.ndarray                          

@dataclass
class BranchPack:
    """Container for all branches, including the default one."""
    default: BranchResult                 
    branches: List[BranchResult]           


# ---- Utility functions ----
def _ensure_square(A: np.ndarray) -> None:
    """Ensure that the matrix A is square."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")

def _matrix_exp(A: np.ndarray, h: float) -> np.ndarray:
    """Compute the matrix exponential P = exp(-h * A)."""
    _ensure_square(A)
    return expm(-h * A)

def _tie_params(base: Params, cond: Cond, mode: str = "left") -> Params:
    """
    Force parameters in cond = (a, b) to become equal according to the specified mode.

    Modes:
      - 'left'  : overwrite b with a's value
      - 'right' : overwrite a with b's value
      - 'avg'   : replace both with their average value
    """
    a, b = cond
    if a not in base or b not in base:
        raise KeyError(f"Unknown parameter in condition: {cond}")
    out = dict(base)
    if mode == "left":
        out[b] = out[a]
    elif mode == "right":
        out[a] = out[b]
    elif mode == "avg":
        v = 0.5 * (out[a] + out[b])
        out[a] = out[b] = v
    else:
        raise ValueError("mode must be one of: 'left', 'right', or 'avg'")
    return out


# ---- Main process----
def build_transition_branches_numeric(
    A_fn: AFunction,
    h: float,
    base_params: Params,
    param_names: Iterable[str],
    conditions: Optional[Iterable[Cond]] = None,
    tie_mode: str = "left",
) -> BranchPack:
    """
    Construct default and conditional transition branches numerically.

    Steps:
      1) Given a system x' = A x, build A using A_fn(base_params).
      2) Compute the default propagator P0 = exp(-h * A).
      3) If no conditions are provided, generate all pairwise equality candidates.
      4) For each condition, make the corresponding parameters equal,
         rebuild A' = A_fn(params'), and compute P' = exp(-h * A').
      5) Collect all branches into a BranchPack object.
    """
    # Step 1–2: default branch
    A0 = np.asarray(A_fn(dict(base_params)), dtype=float)
    P0 = _matrix_exp(A0, h)
    default = BranchResult(condition=None, P=P0)

    # Step 3: generate pairwise equality conditions if none provided
    names = list(param_names)
    if conditions is None:
        conditions = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    # Step 4–5: compute each conditional branch
    branches: List[BranchResult] = []
    for cond in conditions:
        params_ = _tie_params(base_params, cond, mode=tie_mode)
        A_ = np.asarray(A_fn(params_), dtype=float)
        P_ = _matrix_exp(A_, h)
        branches.append(BranchResult(condition=tuple(sorted(cond)), P=P_))

    return BranchPack(default=default, branches=branches)

