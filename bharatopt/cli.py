# bharatopt/cli.py
"""Command‑line interface for BharatOpt.

The CLI is built with **Typer** and uses **Rich** to render a horizontal chain of
`Panel` boxes representing the optimisation pipeline. It also provides a ``solve``
command that runs the bounded‑variable dual simplex on two hard‑coded toy LPs
and streams a live view of the iterations.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
console = Console()
# Local imports for the solver, data, and UI
from .data.toy_lps import ALL_LPS
from .solver.simplex import dual_simplex
from .ui.simplex_view import display_solver
from .solver import branch_and_bound as bnb_module
from .solver.branch_and_bound import branch_and_bound
from .ui.bnb_view import display_bnb

app = typer.Typer(help="BharatOpt – a demonstration CLI that prints the optimisation pipeline and solves toy LPs.")

# Define the pipeline stages in order
STAGES = [
    "Input",
    "Model",
    "Presolve",
    "Solver Selection",
    "LP Engine (Dual Simplex)",
    "Branch & Bound",
    "Verification",
    "Results",
]

def _make_panel(stage: str) -> Panel:
    """Create a Rich Panel for a single pipeline stage.

    The panel uses a subtle gradient style to look premium.
    """
    text = Text(stage, style="bold white on dark_blue")
    return Panel(text, border_style="bright_magenta", padding=(0, 1))

@app.command()
def show_pipeline() -> None:
    """Print the optimisation pipeline as a horizontal chain of panels.

    The panels are joined by a right‑arrow (→) to illustrate data flow.
    """
    console = Console()
    panels = [_make_panel(stage) for stage in STAGES]
    for i, panel in enumerate(panels):
        console.print(panel, end="")
        if i < len(panels) - 1:
            console.print(Text(" -> ", style="bold bright_yellow"), end="")
    console.print()

@app.command()
def solve(lp_name: str = typer.Argument(..., help="Name of the toy problem to solve (lp1, lp2, milp1)"), milp: bool = typer.Option(False, "--milp", help="Solve as MILP using branch‑and‑bound")) -> None:
    """Solve a toy problem.

    If ``milp`` is ``True`` the problem is treated as a mixed‑integer linear program
    and solved with the branch‑and‑bound algorithm; otherwise the bounded‑variable
    dual simplex is used.
    """
    key = lp_name.lower()
    if key not in ALL_LPS:
        typer.echo(f"[red]Unknown problem '{lp_name}'. Available: {', '.join(ALL_LPS)}[/]")
        raise typer.Exit(code=1)
    lp = ALL_LPS[key]
    if milp:
        gen = branch_and_bound(lp)
        display_bnb(lp_name.upper(), gen)
        obj = bnb_module.best_incumbent
    else:
        gen = dual_simplex(lp)
        display_solver(lp_name.upper(), gen, lp)
        final = list(dual_simplex(lp))[-1]
        obj = final["objective"]
    known = lp["optimal"]
    if abs(obj - known) <= 1e-7:
        console.print(f"[green]MATCHES KNOWN OPTIMUM ({known})[/]")
    else:
        console.print(f"[red]MISMATCH — DEBUG (obj={obj}, known={known})[/]")



@app.command()
def solve_milp(milp_name: str = typer.Argument(..., help="Name of the MILP to solve (milp1)")) -> None:
    """Solve a MILP using branch‑and‑bound and display a live UI.

    The generator yields node events and gap values which are visualised by
    ``display_bnb``. After the search finishes the best incumbent objective is
    compared against the known optimum.
    """
    key = milp_name.lower()
    if key not in ALL_LPS:
        typer.echo(f"[red]Unknown MILP '{milp_name}'. Available: {', '.join(ALL_LPS)}[/]")
        raise typer.Exit(code=1)
    lp = ALL_LPS[key]
    gen = branch_and_bound(lp)
    display_bnb(milp_name.upper(), gen)
    # ``best_incumbent`` is updated by the generator; read via module ref to get live value
    obj = bnb_module.best_incumbent
    known = lp["optimal"]
    if abs(obj - known) <= 1e-7:
        console.print(f"[green]MATCHES KNOWN OPTIMUM ({known})[/]")
    else:
        console.print(f"[red]MISMATCH — DEBUG (obj={obj}, known={known})[/]")

if __name__ == "__main__":
    app()
