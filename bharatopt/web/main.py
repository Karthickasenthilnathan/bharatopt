"""FastAPI app exposing BharatOpt solver results as JSON."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Dict, List

import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from bharatopt.data.toy_lps import ALL_LPS, ITC_ALLOCATION
from bharatopt.solver.branch_and_bound import branch_and_bound
from bharatopt.solver import branch_and_bound as bnb_module
from bharatopt.solver.simplex import dual_simplex
from bharatopt.solver.verify import verify_solution

STATIC_DIR = Path(__file__).with_name("static")
SEGMENT_NAMES = [
    "Cigarettes",
    "FMCG-Others",
    "Agri-Business",
    "Paperboards & Packaging",
]

ROADMAP_TABLE = {
    "columns": [
        "implemented",
        "simplified",
        "roadmap",
    ],
    "rows": [
        {
            "implemented": [
                "dual simplex with KKT verification",
                "branch-and-bound with live gap tracking",
            ],
            "simplified": [
                "hand-built toy instances instead of MPS/Netlib parsing",
                "basic branching instead of full best-bound pruning",
            ],
            "roadmap": [
                "MPS/Netlib/MIPLIB support",
                "presolve",
                "GPU-accelerated interior-point engine",
                "Gomory cuts",
                "industrial blending demo",
            ],
        }
    ],
}

PIPELINE_STAGES = [
    "Input",
    "Model",
    "Presolve",
    "Solver Selection",
    "LP Engine (Dual Simplex)",
    "Branch & Bound",
    "Verification",
    "Results",
]

app = FastAPI(title="BharatOpt API")


def _known_optimum(lp: Dict[str, Any]) -> float:
    return float(lp.get("reported_optimal", lp["optimal"]))


def _reported_objective(lp: Dict[str, Any], objective: float) -> float:
    return float(objective * lp.get("report_objective_sign", 1.0))


def _matches_known(lp: Dict[str, Any], objective: float) -> bool:
    known = _known_optimum(lp)
    tolerance = float(lp.get("optimum_tolerance", 1e-7))
    return abs(objective - known) <= tolerance


def _kkt_payload(lp: Dict[str, Any], final_iteration: Dict[str, Any]) -> Dict[str, Dict[str, float | bool]]:
    checks = verify_solution(lp, final_iteration["primal"], final_iteration["dual"])
    return {
        "primal_feasibility": {
            "passed": bool(checks["primal"][0]),
            "value": float(checks["primal"][1]),
        },
        "dual_feasibility": {
            "passed": bool(checks["dual"][0]),
            "value": float(checks["dual"][1]),
        },
        "complementary_slackness": {
            "passed": bool(checks["csl"][0]),
            "value": float(checks["csl"][1]),
        },
    }


def _solve_lp_payload(name: str, lp: Dict[str, Any]) -> Dict[str, Any]:
    iterations = list(dual_simplex(lp))
    if not iterations:
        raise RuntimeError(f"Solver produced no iterations for {name}.")

    final_iteration = iterations[-1]
    objective = _reported_objective(lp, float(final_iteration["objective"]))
    return {
        "iterations": iterations,
        "kkt": _kkt_payload(lp, final_iteration),
        "objective": objective,
        "known_optimum": _known_optimum(lp),
        "matches": _matches_known(lp, objective),
    }


@app.get("/api/solve-lp/{name}")
def solve_lp(name: str) -> Dict[str, Any]:
    key = name.lower()
    if key not in {"lp1", "lp2"}:
        raise HTTPException(status_code=404, detail="Unknown LP. Available: lp1, lp2")
    return _solve_lp_payload(key, ALL_LPS[key])


@app.get("/api/solve-milp/{name}")
def solve_milp(name: str) -> Dict[str, Any]:
    key = name.lower()
    if key != "milp1":
        raise HTTPException(status_code=404, detail="Unknown MILP. Available: milp1")

    lp = ALL_LPS[key]
    nodes: List[Dict[str, Any]] = []
    gap_history: List[Dict[str, float | int | None]] = []

    for item in branch_and_bound(lp):
        if isinstance(item, dict):
            if item.get("node_id") is not None:
                nodes.append(
                    {
                        "node_id": item.get("node_id"),
                        "parent_id": item.get("parent_id"),
                        "depth": item.get("depth"),
                        "lp_bound": item.get("lp_bound"),
                        "status": item.get("status"),
                    }
                )
        else:
            gap = float(item)
            gap_history.append(
                {
                    "node_count": len(nodes),
                    "gap_pct": gap * 100.0 if isfinite(gap) else None,
                }
            )

    objective = float(bnb_module.best_incumbent)
    known = float(lp["optimal"])
    return {
        "nodes": nodes,
        "gap_history": gap_history,
        "objective": objective,
        "known_optimum": known,
        "matches": abs(objective - known) <= 1e-7,
    }


@app.get("/api/itc-allocation")
def itc_allocation() -> Dict[str, Any]:
    payload = _solve_lp_payload("itc_allocation", ITC_ALLOCATION)
    final_iteration = payload["iterations"][-1]
    return {
        "allocation": dict(zip(SEGMENT_NAMES, final_iteration["primal"])),
        "objective": payload["objective"],
        "known_optimum": payload["known_optimum"],
        "matches": payload["matches"],
        "kkt": payload["kkt"],
        "disclaimer_text": ITC_ALLOCATION.get("disclaimer", ""),
    }


@app.get("/api/roadmap")
def roadmap() -> Dict[str, Any]:
    return ROADMAP_TABLE


@app.get("/api/pipeline")
def pipeline() -> Dict[str, List[Dict[str, int | str]]]:
    return {
        "stages": [
            {"index": index, "name": stage}
            for index, stage in enumerate(PIPELINE_STAGES, start=1)
        ]
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    url = "http://localhost:8000"
    typer.echo(f"BharatOpt web server starting at {url}")
    uvicorn.run("bharatopt.web.main:app", host="localhost", port=8000)


if __name__ == "__main__":
    main()

