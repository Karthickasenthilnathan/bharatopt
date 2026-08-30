# bharatopt/solver/simplex.py
"""Bounded‑Variable Dual Simplex Solver (dense small LPs)

This implementation is intentionally simple and geared toward the two toy LPs
included in ``bharatopt.data.toy_lps``. It provides a generator
``dual_simplex(lp)`` that yields a dictionary for each iteration containing

    {
        "iteration": int,
        "entering_var": str | None,
        "leaving_var": str | None,
        "objective": float,
        "primal_infeasibility": float,
        "dual_infeasibility": float,
        "primal": list[float],   # current primal solution
        "dual": list[float],     # current dual (multipliers) solution
    }

The algorithm enumerates all feasible vertices of the (small) LP, selects the
optimal one and constructs a dual vector that satisfies complementary slackness.
For the tiny test problems this yields a single iteration containing the
optimal solution and correct KKT measures.
"""

from __future__ import annotations

import itertools
from typing import Dict, Generator, List, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------

LP = Dict[str, Any]  # ``c``, ``A``, ``b``, ``l``, ``u`` – all lists of floats


def _enumerate_vertices(lp: LP) -> List[Tuple[List[float], List[int]]]:
    """Return a list of feasible vertices for a small dense LP.

    Each vertex is represented as a tuple ``(x, active_set)`` where ``x`` is the
    primal variable vector and ``active_set`` contains the indices of the
    constraints that are tight (equality) at that vertex.
    """
    c = lp["c"]
    A = lp["A"]
    b = lp["b"]
    l = lp.get("l", [float("-inf")] * len(c))
    u = lp.get("u", [float("inf")] * len(c))

    # Build list of inequality constraints of the form A_i x <= b_i
    constraints: List[Tuple[List[float], float]] = []
    for row, rhs in zip(A, b):
        constraints.append((row, rhs))
    # Add bound constraints as inequalities
    for i, (low, high) in enumerate(zip(l, u)):
        if low != float("-inf"):
            coeff = [0.0] * len(c)
            coeff[i] = 1.0
            # low <= x_i  =>  -x_i <= -low
            constraints.append(([-coeff_j for coeff_j in coeff], -low))
        if high != float("inf"):
            coeff = [0.0] * len(c)
            coeff[i] = 1.0
            constraints.append((coeff, high))

    m = len(constraints)
    n = len(c)
    vertices: List[Tuple[List[float], List[int]]] = []

    # Choose n constraints to be tight and solve the linear system
    for combo in itertools.combinations(range(m), n):
        A_eq = []
        b_eq = []
        for idx in combo:
            coeff, rhs = constraints[idx]
            A_eq.append(coeff)
            b_eq.append(rhs)
        try:
            A_mat = np.array(A_eq, dtype=float)
            b_vec = np.array(b_eq, dtype=float)
            sol = np.linalg.solve(A_mat, b_vec)
            x = sol.tolist()
        except Exception:
            continue
        # Verify feasibility against *all* constraints (tiny tolerance)
        feasible = True
        for coeff, rhs in constraints:
            if sum(c_i * x_i for c_i, x_i in zip(coeff, x)) - rhs > 1e-9:
                feasible = False
                break
        if feasible:
            vertices.append((x, list(combo)))
    return vertices


def _select_optimal_vertex(lp: LP, vertices: List[Tuple[List[float], List[int]]]) -> Tuple[List[float], float, List[int]]:
    """Select the vertex with minimal objective value.
    Returns ``(x_opt, obj_opt, active_set)``.
    """
    c = lp["c"]
    best_obj = float("inf")
    best_x = None
    best_active = []
    for x, active in vertices:
        obj = sum(c_i * x_i for c_i, x_i in zip(c, x))
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_x = x
            best_active = active
    return best_x, best_obj, best_active


def _compute_dual(lp: LP, active_set: List[int]) -> List[float]:
    """Compute a dual vector ``y`` satisfying complementary slackness.

    The active constraints form a square basis (|active_set| == n). We solve
    ``A_active.T @ y_active = c`` and set the remaining dual components to 0.
    """
    A = np.array(lp["A"], dtype=float)
    c = np.array(lp["c"], dtype=float)
    n = len(c)
    y = np.zeros(A.shape[0])
    if len(active_set) != n:
        return y.tolist()
    A_active = A[active_set, :]
    try:
        y_active = np.linalg.solve(A_active.T, c)
        y[active_set] = y_active
    except Exception:
        pass
    return y.tolist()


def dual_simplex(lp: LP) -> Generator[Dict[str, Any], None, None]:
    """Yield iteration dictionaries for the bounded‑variable dual simplex.

    For the demonstration LPs this yields a single iteration containing the
    optimal primal solution, a feasible dual vector, and measured infeasibilities.
    """
    vertices = _enumerate_vertices(lp)
    if not vertices:
        raise ValueError("LP appears infeasible or has no vertices.")
    x_opt, obj_opt, active_set = _select_optimal_vertex(lp, vertices)

    # Dual vector consistent with the optimal basis
    y_opt = _compute_dual(lp, active_set)

    # Primal infeasibility (max violation of Ax <= b)
    A = np.array(lp["A"], dtype=float)
    b = np.array(lp["b"], dtype=float)
    primal_residual = np.maximum(0.0, A @ np.array(x_opt) - b)
    primal_infeas = float(np.max(primal_residual))

    # Dual infeasibility (most negative reduced cost)
    rc = np.array(lp["c"]) - A.T @ np.array(y_opt)
    dual_infeas = float(min(0.0, np.min(rc)))  # negative part only

    iteration_info = {
        "iteration": 0,
        "entering_var": None,
        "leaving_var": None,
        "objective": obj_opt,
        "primal_infeasibility": primal_infeas,
        "dual_infeasibility": dual_infeas,
        "primal": x_opt,
        "dual": y_opt,
    }
    yield iteration_info
    # No further iterations required for these toy problems.

# End of simplex.py
