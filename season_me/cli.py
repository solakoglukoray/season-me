"""Typer CLI for season-me."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .main import analyze

_HELP = "Upload a selfie → discover your personal color season and flattering palette."
app = typer.Typer(
    name="season-me",
    help=_HELP,
    add_completion=False,
)
console = Console(highlight=False, emoji=False)

_SEASON_STYLES: dict[str, str] = {
    "Spring": "bold yellow",
    "Summer": "bold cyan",
    "Autumn": "bold red",
    "Winter": "bold blue",
}


@app.command()
def run(
    image: Annotated[
        Path,
        typer.Argument(
            help="Path to a selfie or portrait photo (JPG/PNG).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show raw skin tone HSL metrics.",
        ),
    ] = False,
) -> None:
    """Analyze a selfie and find your personal color season."""
    console.print(f"\n[dim]Analyzing[/dim] [bold]{image.name}[/bold]...\n")

    try:
        result = analyze(image)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    style = _SEASON_STYLES.get(result.season, "bold white")
    border = style.replace("bold ", "")

    console.print(
        Panel(
            f"[{style}]{result.season}[/{style}]\n\n{result.description}",
            title="Your Color Season",
            border_style=border,
            expand=False,
        )
    )

    console.print(f"\n[dim]Detected skin tone:[/dim] {result.skin_tone_hex}")

    if verbose:
        console.print(
            f"  CIE Lab: L*={result.lab_L:.1f}  "
            f"a*={result.lab_a:.2f}  "
            f"b*={result.lab_b:.2f}"
        )

    table = Table(
        title=f"{result.season} Palette",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Swatch", justify="center", width=8)
    table.add_column("Color Name")
    table.add_column("Hex Code")

    for entry in result.palette:
        swatch = Text("  ##  ", style=entry["hex"])
        table.add_row(swatch, entry["name"], entry["hex"])

    console.print()
    console.print(table)
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
