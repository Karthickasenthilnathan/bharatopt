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

# ITC_ALLOCATION: maximise capital allocation across ITC Limited segments.
#
# Source: BharatOpt_RealWorld_Profit_Dataset_ITC.pdf
# Real financial data from FY2021-FY2025 and Q2 FY26 segment results.
# The solver minimises c^T x, so the revenue-share maximisation coefficients
# are stored negated and reported back with ``report_objective_sign``.
ITC_ALLOCATION: LP = {
    "c": [-0.410, -0.263, -0.176, -0.067],
    "A": [
        [1.0, 1.0, 1.0, 1.0],      # total capital budget
        [1.0, 0.0, 0.0, 0.0],      # Cigarettes capacity proxy
        [0.0, 1.0, 0.0, 0.0],      # FMCG-Others capacity proxy
        [0.0, 0.0, 1.0, 0.0],      # Agri-Business capacity proxy
        [0.0, 0.0, 0.0, 1.0],      # Paperboards & Packaging capacity proxy
        [-1.0, 0.0, 0.0, 0.0],     # Cigarettes non-negativity
        [0.0, -1.0, 0.0, 0.0],     # FMCG-Others non-negativity
        [0.0, 0.0, -1.0, 0.0],     # Agri-Business non-negativity
        [0.0, 0.0, 0.0, -1.0],     # Paperboards & Packaging non-negativity
    ],
    "b": [
        69966.6667,
        37657.36,
        24236.48,
        16151.20,
        6144.00,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "l": [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
    "u": [float("inf"), float("inf"), float("inf"), float("inf")],
    "optimal": -23234.529333,
    "reported_optimal": 23234.529333,
    "report_objective_sign": -1.0,
    "optimum_tolerance": 1e-4,
    "verified_label": "MATCHES VERIFIED OPTIMUM",
    "disclaimer": (
        "Real financial data (ITC Ltd, FY2021-FY2025 + Q2 FY26 segment results). "
        "Capital budget derived from ROCE; "
    ),
}

ALL_LPS = {
    "lp1": LP1,
    "lp2": LP2,
    "milp1": MILP1,
    "itc_allocation": ITC_ALLOCATION,
}
