"""bharatopt/ui/roadmap_view.py

Rich UI view presenting the project roadmap.

The view prints a table with three columns:
- Implemented (real, running now)
- Simplified for this sprint
- Full roadmap (next phase)

Each column is populated with the current status and future plans.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table


def display_roadmap() -> None:
    """Render the BharatOpt roadmap as a Rich table.

    The table has three columns describing the current implementation,
    the simplified sprint version, and the longer‑term roadmap.
    """
    console = Console()
    table = Table(
        title="BharatOpt Roadmap",
        expand=True,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column(
        "Implemented (real, running now)",
        style="bold green",
        justify="left",
    )
    table.add_column(
        "Simplified for this sprint",
        style="cyan",
        justify="left",
    )
    table.add_column(
        "Full roadmap (next phase)",
        style="yellow",
        justify="left",
    )

    implemented = (
        "dual simplex with KKT verification,\n"
        "branch‑and‑bound with live gap tracking"
    )
    simplified = (
        "hand‑built toy instances instead of MPS/Netlib parsing,\n"
        "basic branching instead of full best‑bound pruning"
    )
    roadmap = (
        "MPS/Netlib/MIPLIB support, presolve, GPU‑accelerated interior‑point engine,\n"
        "Gomory cuts, industrial blending demo"
    )
    table.add_row(implemented, simplified, roadmap)
    console.print(table)

if __name__ == "__main__":
    display_roadmap()
