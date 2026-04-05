"""
llm-extract CLI

Usage:
    llm-extract path/to/document.pdf --schema schemas/invoice.py
    llm-extract document.txt --fields "name,date,amount" --out result.json
    llm-extract document.pdf --schema schemas/contract.py --show-sources
    llm-extract --formats
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Optional

import typer
from pydantic import BaseModel
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="llm-extract",
    help="Extract structured data from any document using LLMs.",
    add_completion=False,
)
console = Console()


@app.command()
def extract_cmd(
    source: str = typer.Argument(
        ..., help="Path to document (PDF, DOCX, HTML, CSV, TXT, MD, JSON)"
    ),
    schema_file: Optional[Path] = typer.Option(
        None, "--schema", "-s", help="Python file containing a Pydantic schema class"
    ),
    fields: Optional[str] = typer.Option(
        None,
        "--fields",
        "-f",
        help="Comma-separated field names for quick extraction (no schema file needed)",
    ),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Save result as JSON file"),
    show_sources: bool = typer.Option(
        False, "--show-sources", help="Show source text snippets for each field"
    ),
    show_raw: bool = typer.Option(
        False, "--show-raw", help="Show raw LLM response (for debugging)"
    ),
    model: str = typer.Option("claude-opus-4-5", "--model", "-m", help="Anthropic model to use"),
    chunk_size: int = typer.Option(8000, "--chunk-size", help="Max characters per chunk"),
    confidence_threshold: float = typer.Option(
        0.0, "--min-confidence", help="Minimum confidence to include field"
    ),
    instructions: Optional[str] = typer.Option(
        None, "--instructions", "-i", help="Additional extraction instructions"
    ),
) -> None:
    """Extract structured data from a document file."""
    from llm_extract.extractor import extract
    from llm_extract.loader import detect_format
    from llm_extract.models import ExtractionConfig

    console.print()
    fmt = detect_format(source)
    console.print(
        Panel(
            f"[bold]Document:[/bold] {source} ({fmt})\n"
            f"[bold]Model:[/bold] {model}\n"
            f"[dim]Extracting structured data...[/dim]",
            title="🔍 llm-extract",
            border_style="cyan",
        )
    )

    # Build schema
    schema_cls = _resolve_schema(schema_file, fields)
    if schema_cls is None:
        console.print("[red]Error: provide --schema or --fields[/red]")
        raise typer.Exit(1)

    config = ExtractionConfig(
        model=model,
        chunk_size=chunk_size,
        confidence_threshold=confidence_threshold,
    )

    # Run extraction
    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True
    ) as p:
        p.add_task("Extracting...", total=None)
        result = extract(source, schema=schema_cls, config=config, instructions=instructions)

    # Display result
    _display_result(
        result, show_sources=show_sources, show_raw=show_raw, threshold=confidence_threshold
    )

    # Save if requested
    if out and result.success:
        output_data = {
            "data": result.data.model_dump(),
            "confidence": result.confidence,
            "sources": result.sources if show_sources else {},
            "coverage": result.coverage,
            "mean_confidence": result.mean_confidence,
        }
        out.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[green]💾 Saved to: {out}[/green]")


@app.command("formats")
def list_formats() -> None:
    """List all supported document formats."""
    table = Table(title="Supported Document Formats", box=box.ROUNDED)
    table.add_column("Extension")
    table.add_column("Format")
    table.add_column("Extra Dependency")

    formats = [
        (".txt, .md, .rst", "Plain text / Markdown", "None (built-in)"),
        (".csv", "CSV (converted to table)", "None (built-in)"),
        (".json", "JSON", "None (built-in)"),
        (".html, .htm", "HTML (tags stripped)", "pip install 'llm-extract[html]'"),
        (".pdf", "PDF", "pip install 'llm-extract[pdf]'"),
        (".docx", "Word Document", "pip install 'llm-extract[docx]'"),
    ]
    for ext, fmt, dep in formats:
        table.add_row(ext, fmt, dep)
    console.print(table)


@app.command("schema")
def show_schema(
    schema_file: Path = typer.Argument(..., help="Python file with Pydantic schema")
) -> None:
    """Inspect a schema file and show extraction fields."""
    schema_cls = _load_schema_from_file(schema_file)
    if schema_cls is None:
        console.print(f"[red]No Pydantic BaseModel found in {schema_file}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Schema: {schema_cls.__name__}", box=box.ROUNDED)
    table.add_column("Field")
    table.add_column("Type")
    table.add_column("Required", justify="center")
    table.add_column("Description")

    for name, info in schema_cls.model_fields.items():
        ann = str(getattr(info.annotation, "__name__", info.annotation))
        req = "✓" if info.is_required() else "○"
        desc = info.description or ""
        table.add_row(name, ann, req, desc)

    console.print(table)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_schema(
    schema_file: Optional[Path],
    fields: Optional[str],
) -> type[BaseModel] | None:
    """Resolve schema from file or quick-fields string."""
    if schema_file:
        return _load_schema_from_file(schema_file)
    if fields:
        return _build_quick_schema(fields)
    return None


def _load_schema_from_file(path: Path) -> type[BaseModel] | None:
    """Load a Pydantic BaseModel subclass from a Python file."""
    spec = importlib.util.spec_from_file_location("_schema_module", path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first BaseModel subclass (excluding BaseModel itself)
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
                return obj
        except TypeError:
            continue
    return None


def _build_quick_schema(fields_str: str) -> type[BaseModel]:
    """Build a simple schema from a comma-separated field list."""
    from pydantic import create_model

    field_names = [f.strip() for f in fields_str.split(",") if f.strip()]
    field_definitions = {name: (Optional[str], None) for name in field_names}
    return create_model("QuickSchema", **field_definitions)


def _display_result(result, show_sources: bool, show_raw: bool, threshold: float) -> None:
    """Display extraction result in the terminal."""
    if not result.success:
        console.print("\n[red]✗ Extraction failed[/red]")
        if result.raw_response:
            console.print(f"[dim]{result.raw_response[:500]}[/dim]")
        return

    console.print(
        f"\n[green]✓ Extracted {result.fields_found}/{result.fields_total} fields "
        f"(coverage: {result.coverage:.0%}, mean confidence: {result.mean_confidence:.2f})[/green]"
    )

    if result.chunks_used > 1:
        console.print(f"[dim]Processed {result.chunks_used} chunks[/dim]")

    # Results table
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Field", style="white")
    table.add_column("Value", style="cyan")
    table.add_column("Confidence", justify="right")
    if show_sources:
        table.add_column("Source", style="dim")

    data_dict = result.data.model_dump()
    for field_name, value in data_dict.items():
        conf = result.confidence.get(field_name, 0.0)
        if conf < threshold and value is not None:
            continue
        conf_color = "green" if conf >= 0.9 else "yellow" if conf >= 0.7 else "red"
        conf_str = (
            f"[{conf_color}]{conf:.2f}[/{conf_color}]" if value is not None else "[dim]—[/dim]"
        )
        value_str = str(value) if value is not None else "[dim]not found[/dim]"
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."

        if show_sources:
            source = result.sources.get(field_name, "")
            if len(source) > 50:
                source = source[:47] + "..."
            table.add_row(field_name, value_str, conf_str, source)
        else:
            table.add_row(field_name, value_str, conf_str)

    console.print(table)

    low_conf = result.low_confidence_fields(threshold=0.7)
    if low_conf:
        console.print(f"\n[yellow]⚠ Low confidence fields: {', '.join(low_conf)}[/yellow]")

    if show_raw:
        console.print(f"\n[dim]Raw response:[/dim]\n{result.raw_response[:1000]}")


def main():
    app()


if __name__ == "__main__":
    main()
