# bharatopt/ui/bnb_view.py
"""Rich live view for the best‑bound branch‑and‑bound MILP solver.

The generator from ``branch_and_bound`` yields alternating items:
    1) A dict describing a processed node (event)
    2) A float representing the current optimality gap.
The UI consumes these items, updates a scrolling log of events and a single‑
progress bar that shows ``gap * 100`` percent (counting down to 0%).

Node status colours:
    * ``branched``   – cyan
    * ``pruned``     – dim
    * ``incumbent``  – bold green
    * ``finished``   – magenta (final marker)
"""

from __future__ import annotations

from typing import Generator, Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text


def _build_event_table() -> Table:
    table = Table(expand=True, show_header=True, header_style="bold magenta")
    table.add_column("Node ID", justify="right")
    table.add_column("Parent ID", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("LP Bound", justify="right")
    table.add_column("Status", style="bold")
    return table


def display_bnb(lp_name: str, gen: Generator[Any, None, None]) -> None:
    """Render a live view of B&B events and the current gap.

    Parameters
    ----------
    lp_name:
        Human‑readable identifier for the problem (e.g. ``"MILP1"``).
    gen:
        Generator produced by ``branch_and_bound(lp)``.
    """
    console = Console()
    table = _build_event_table()
    progress = Progress(
        TextColumn("[progress.description]Gap %:"),
        BarColumn(bar_width=None),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        expand=True,
    )
    gap_task = progress.add_task("gap", total=100.0, completed=100.0)

    with Live(Panel.fit(table, title=f"B&B Progress – {lp_name}"), refresh_per_second=6, console=console) as live:
        # Render progress bar below the table using a separate Live instance
        with Live(progress, refresh_per_second=6, console=console) as prog_live:
            for item in gen:
                if isinstance(item, dict):
                    ev = item
                    # colour based on status
                    status = ev.get("status", "")
                    if status == "branched":
                        style = "cyan"
                    elif status == "pruned":
                        style = "dim"
                    elif status == "incumbent":
                        style = "bold green"
                    elif status == "finished":
                        style = "magenta"
                    else:
                        style = ""
                    row = [
                        str(ev.get("node_id", "-")),
                        str(ev.get("parent_id", "-")),
                        str(ev.get("depth", "-")),
                        f"{ev.get('lp_bound', '-'):.4f}" if ev.get("lp_bound") is not None else "-",
                        Text(status, style=style),
                    ]
                    table.add_row(*row)
                    live.update(Panel.fit(table, title=f"B&B Progress – {lp_name}"))
                else:
                    # item is a gap float
                    gap = float(item)
                    pct = max(0.0, min(100.0, gap * 100.0))
                    progress.update(gap_task, completed=100.0 - pct)
                    prog_live.refresh()
    # Final newline for cleanliness
    console.print()

# End of bnb_view.py
