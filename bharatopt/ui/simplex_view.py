# bharatopt/ui/simplex_view.py
"""Rich live view for the bounded‑variable dual simplex solver.

The :func:`display_solver` function accepts a generator produced by
``dual_simplex(lp)`` and streams a table of iteration information.  After the
generator exhausts, it runs the KKT verification and renders a panel showing
PASS/FAIL for each condition.
"""

from __future__ import annotations

from typing import Generator, Dict, Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..solver.verify import verify_solution


def _build_iteration_table() -> Table:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True, header_style="bold magenta")
    table.add_column("Iter", justify="right")
    table.add_column("Entering Var", style="cyan")
    table.add_column("Leaving Var", style="cyan")
    table.add_column("Objective", justify="right")
    table.add_column("Primal infeas.", justify="right")
    table.add_column("Dual infeas.", justify="right")
    return table


def _format_pass_fail(passed: bool) -> Text:
    if passed:
        return Text("PASS", style="bold green")
    else:
        return Text("FAIL", style="bold red")


def display_solver(lp_name: str, gen: Generator[Dict[str, Any], None, None], lp: Dict[str, Any]) -> None:
    """Render live iteration table and final KKT panel.

    Parameters
    ----------
    lp_name:
        Human readable identifier (e.g. "LP1").
    gen:
        Generator yielded by :func:`dual_simplex`.
    lp:
        Original LP dictionary – needed for verification.
    """
    console = Console()
    table = _build_iteration_table()
    with Live(table, refresh_per_second=4, console=console, vertical_overflow="visible"):
        final_iter = None
        for it in gen:
            final_iter = it
            table.add_row(
                str(it["iteration"]),
                str(it.get("entering_var") or "-"),
                str(it.get("leaving_var") or "-"),
                f"{it['objective']:.6f}",
                f"{it['primal_infeasibility']:.2e}",
                f"{it['dual_infeasibility']:.2e}",
            )
    # After live view ends, show verification results
    if final_iter is None:
        console.print("[bold red]Solver produced no iterations.[/]")
        return
    ver = verify_solution(lp, final_iter["primal"], final_iter["dual"])
    kkt_table = Table.grid(padding=(0, 1))
    kkt_table.add_column(justify="right")
    kkt_table.add_column()
    kkt_table.add_row("Primal feasibility", _format_pass_fail(ver["primal"][0]))
    kkt_table.add_row("Dual feasibility", _format_pass_fail(ver["dual"][0]))
    kkt_table.add_row("Complementary slackness", _format_pass_fail(ver["csl"][0]))
    panel = Panel(
        kkt_table,
        title=f"KKT Verification for {lp_name}",
        border_style="bright_blue",
    )
    console.print(panel)

# End of simplex_view.py
