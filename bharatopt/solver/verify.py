# bharatopt/solver/verify.py
"""Verification utilities for LP solutions produced by the dual simplex.

The functions compute three classic KKT checks:

1. **Primal feasibility** – infinity norm of the residual ``|Ax - b|_∞``.
2. **Dual feasibility** – sign of reduced costs (for minimisation ``c - Aᵀy ≥ 0``).
3. **Complementary slackness** – product of primal variable distance to its bound
   and the corresponding dual activity should be zero.

Each check returns a tuple ``(passed: bool, value: float)`` where ``value`` is the
measure (e.g. residual norm).  The helper ``verify_solution`` aggregates the three
checks and returns a dict suitable for the UI.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Any, List

LP = Dict[str, Any]

TOL = 1e-7


def primal_residual(lp: LP, x: List[float]) -> Tuple[bool, float]:
    """Return ``(pass, ‖Ax - b‖_∞)`` for the primal feasibility check."""
    A = np.array(lp["A"], dtype=float)
    b = np.array(lp["b"], dtype=float)
    x_arr = np.array(x, dtype=float)
    residual = np.maximum(0.0, A @ x_arr - b)
    norm = float(np.max(residual))
    return (norm <= TOL, norm)


def dual_residual(lp: LP, y: List[float]) -> Tuple[bool, float]:
    """Return ``(pass, max_negative_reduced_cost)``.

    For a minimisation problem the dual feasibility condition is ``c - Aᵀy ≥ 0``.
    The function returns the most negative reduced cost (or 0 if all are non‑negative).
    """
    A = np.array(lp["A"], dtype=float)
    c = np.array(lp["c"], dtype=float)
    y_arr = np.array(y, dtype=float)
    rc = c - A.T @ y_arr
    most_negative = np.min(rc)
    return (most_negative >= -TOL, most_negative)


def complementary_slackness(lp: LP, x: List[float], y: List[float]) -> Tuple[bool, float]:
    """Return ``(pass, max|x_i * rc_i|)`` as a simple CSL violation measure.

    The product of a primal variable and its reduced cost should be zero at an
    optimal solution.  We compute the maximum absolute product as a violation
    metric.
    """
    A = np.array(lp["A"], dtype=float)
    c = np.array(lp["c"], dtype=float)
    y_arr = np.array(y, dtype=float)
    rc = c - A.T @ y_arr
    x_arr = np.array(x, dtype=float)
    violation = np.max(np.abs(x_arr * rc))
    return (violation <= TOL, violation)


def verify_solution(lp: LP, x: List[float], y: List[float]) -> Dict[str, Tuple[bool, float]]:
    """Run all three KKT checks and return a dictionary.

    The returned mapping has keys ``"primal"``, ``"dual"`` and ``"csl"`` each
    pointing to a ``(passed, value)`` tuple.
    """
    primal_ok, primal_val = primal_residual(lp, x)
    dual_ok, dual_val = dual_residual(lp, y)
    csl_ok, csl_val = complementary_slackness(lp, x, y)
    return {
        "primal": (primal_ok, primal_val),
        "dual": (dual_ok, dual_val),
        "csl": (csl_ok, csl_val),
    }

# End of verify.py
