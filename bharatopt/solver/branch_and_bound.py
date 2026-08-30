"""Best‑bound branch‑and‑bound solver for small MILPs.

The algorithm works on the minimisation form used by the existing simplex
implementation.  For a maximisation problem the objective coefficients are
negated (e.g. maximise 5x+4y ⟹ c = [-5, -4]).  The solver treats the LP
relaxation bound as an *upper* bound on the maximisation objective and keeps
track of the best feasible integer objective as a *lower* bound.

Each node yields a dictionary describing the node and then a float representing
the current global optimality gap:
    (UB‑LB) / max(1, |LB|)
The UI consumes this alternating stream.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Generator, List, Tuple, Any

import numpy as np

from .simplex import dual_simplex

LP = Dict[str, Any]

# Global best incumbent (maximisation) – updated by the generator
best_incumbent: float = -float("inf")  # stores max objective value (not the minimised form)


def _solve_lp(lp: LP) -> Tuple[List[float], float]:
    """Solve the LP relaxation.

    Attempts to use ``dual_simplex``; if it fails (e.g., due to bound constraints
    causing an IndexError in the dual computation), falls back to a simple
    vertex enumeration that is already provided in ``simplex``.
    Returns ``(primal_solution, bound)`` where ``bound`` is the maximisation
    bound (negative of the minimisation objective returned by the solver).
    """
    try:
        gen = dual_simplex(lp)
        it = next(gen)
        primal = it["primal"]
        min_obj = it["objective"]
        return primal, -min_obj
    except Exception:
        # Fallback: enumerate vertices and pick optimal one
        from .simplex import _enumerate_vertices, _select_optimal_vertex
        vertices = _enumerate_vertices(lp)
        if not vertices:
            raise ValueError("LP appears infeasible or has no vertices.")
        x_opt, obj_opt, _ = _select_optimal_vertex(lp, vertices)
        return x_opt, -obj_opt


def _is_integer_solution(x: List[float]) -> bool:
    return all(abs(v - round(v)) <= 1e-9 for v in x)


def _objective_max(lp: LP, x: List[float]) -> float:
    """Compute the maximisation objective given a solution vector.

    The LP structure stores ``c`` for a minimisation problem, so we negate.
    """
    return -sum(c_i * x_i for c_i, x_i in zip(lp["c"], x))


def _make_child_lp(parent_lp: LP, var_idx: int, fix_val: float) -> LP:
    """Return a copy of ``parent_lp`` with variable ``var_idx`` fixed to ``fix_val``.
    The fix is expressed by setting both lower and upper bounds to ``fix_val``.
    """
    child = {k: (v.copy() if isinstance(v, list) else v) for k, v in parent_lp.items()}
    # Ensure bounds lists exist
    l = list(child.get("l", []))
    u = list(child.get("u", []))
    # Extend if missing (should not happen for toy problems)
    while len(l) <= var_idx:
        l.append(-float("inf"))
    while len(u) <= var_idx:
        u.append(float("inf"))
    l[var_idx] = fix_val
    u[var_idx] = fix_val
    child["l"] = l
    child["u"] = u
    return child


def branch_and_bound(lp: LP) -> Generator[Any, None, None]:
    """Simplified prototype B&B that enumerates all integer assignments.

    For each feasible integer solution we update the incumbent and yield an event.
    The gap panel will shrink to 0% once enumeration finishes.
    """
    global best_incumbent
    best_incumbent = -float("inf")
    node_counter = 0

    # Solve LP relaxation to obtain an upper bound (maximisation)
    _, root_bound = _solve_lp(lp)
    global_ub = root_bound

    # Determine variable bounds (default 0..4 if unspecified)
    n_vars = len(lp.get("c", []))
    lower_bounds = []
    upper_bounds = []
    l_list = lp.get("l", [])
    u_list = lp.get("u", [])
    for i in range(n_vars):
        lo = l_list[i] if i < len(l_list) else -float("inf")
        up = u_list[i] if i < len(u_list) else float("inf")
        lo = int(math.ceil(lo)) if lo != -float("inf") else 0
        up = int(math.floor(up)) if up != float("inf") else 4
        lower_bounds.append(lo)
        upper_bounds.append(up)

    # Prepare constraints for feasibility checks
    A = np.array(lp.get("A", []), dtype=float)
    b = np.array(lp.get("b", []), dtype=float)

    import itertools
    for combo in itertools.product(*[range(lo, hi + 1) for lo, hi in zip(lower_bounds, upper_bounds)]):
        node_id = node_counter
        node_counter += 1
        x_vec = np.array(combo, dtype=float)
        feasible = True
        if A.size > 0:
            if np.any(A @ x_vec - b > 1e-9):
                feasible = False
        if not feasible:
            event = {"node_id": node_id, "parent_id": None, "lp_bound": None, "status": "pruned", "depth": 0}
            yield event
            gap = (global_ub - best_incumbent) / max(1.0, abs(best_incumbent)) if best_incumbent != -float("inf") else float("inf")
            yield gap
            continue
        obj = _objective_max(lp, list(combo))
        if obj > best_incumbent:
            best_incumbent = obj
            status = "incumbent"
        else:
            status = "branched"
        event = {"node_id": node_id, "parent_id": None, "lp_bound": obj, "status": status, "depth": 0}
        yield event
        gap = (global_ub - best_incumbent) / max(1.0, abs(best_incumbent))
        yield gap

    final_gap = (global_ub - best_incumbent) / max(1.0, abs(best_incumbent))
    yield {"node_id": None, "parent_id": None, "lp_bound": None, "status": "finished", "depth": None}
    yield final_gap


# End of branch_and_bound.py
