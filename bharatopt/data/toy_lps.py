# bharatopt/data/toy_lps.py
"""Toy linear programming instances for demonstration.
Each LP is represented as a dictionary compatible with ``bharatopt.solver.simplex``.
The keys are:
    c: list of objective coefficients (minimise cᵀx)
    A: list of constraint rows (each a list of coefficients)
    b: list of right‑hand side values
    l: lower bounds for variables (use -inf for unbounded below)
    u: upper bounds for variables (use inf for unbounded above)
"""

import math
from typing import Dict, List, Any

LP = Dict[str, Any]

# LP1: minimise -x - y
# Constraints (all <= form):
#   x + y <= 4
#   x <= 3
#   y <= 3
#   -x <= 0   (x >= 0)
#   -y <= 0   (y >= 0)
LP1: LP = {
    "c": [-1.0, -1.0],
    "A": [
        [1.0, 1.0],   # x + y <= 4
        [1.0, 0.0],   # x <= 3
        [0.0, 1.0],   # y <= 3
        [-1.0, 0.0],  # -x <= 0  => x >= 0
        [0.0, -1.0],  # -y <= 0  => y >= 0
    ],
    "b": [4.0, 3.0, 3.0, 0.0, 0.0],
    "l": [float("-inf"), float("-inf")],  # bounds are encoded via constraints above
    "u": [float("inf"), float("inf")],
    "optimal": -4.0,
}

# LP2: minimise 2x + 3y
# Constraints (all <= form):
#   -x - y <= -4   (x + y >= 4)
#   -x - 2y <= -5  (x + 2y >= 5)
#   -x <= 0        (x >= 0)
#   -y <= 0        (y >= 0)
LP2: LP = {
    "c": [2.0, 3.0],
    "A": [
        [-1.0, -1.0],   # -x - y <= -4
        [-1.0, -2.0],   # -x - 2y <= -5
        [-1.0, 0.0],    # -x <= 0  => x >= 0
        [0.0, -1.0],    # -y <= 0  => y >= 0
    ],
    "b": [-4.0, -5.0, 0.0, 0.0],
    "l": [float("-inf"), float("-inf")],
    "u": [float("inf"), float("inf")],
    "optimal": 9.0,
}

# Export list for convenience

# MILP1: maximise 5x + 4y
# Constraints (all <= form):
#   6x + 4y <= 24
#   x + 2y <= 6
#   x <= 4
#   y <= 4
#   -x <= 0   (x >= 0)
#   -y <= 0   (y >= 0)
MILP1 = {
    "c": [-5.0, -4.0],  # minimise -5x -4y => maximise 5x + 4y
    "A": [
        [6.0, 4.0],   # 6x + 4y <= 24
        [1.0, 2.0],   # x + 2y <= 6
        [1.0, 0.0],   # x <= 4
        [0.0, 1.0],   # y <= 4
        [-1.0, 0.0],  # -x <= 0  => x >= 0
        [0.0, -1.0],  # -y <= 0  => y >= 0
    ],
    "b": [24.0, 6.0, 4.0, 4.0, 0.0, 0.0],
    "l": [float("-inf"), float("-inf")],  # unbounded below (handled via constraints)
    "u": [float("inf"), float("inf")],   # unbounded above (handled via constraints)
    "optimal": 20.0,
}

ALL_LPS = {"lp1": LP1, "lp2": LP2, "milp1": MILP1}
