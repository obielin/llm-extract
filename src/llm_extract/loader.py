"""
Document loader — converts any supported file format to plain text.

Supported formats:
    .txt, .md, .rst     — plain text, read directly
    .csv                — tabular data, converted to markdown table
    .json               — JSON, pretty-printed
    .html, .htm         — HTML, tags stripped (requires beautifulsoup4)
    .pdf                — PDF text extraction (requires pdfplumber)
    .docx               — Word documents (requires python-docx)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_document(path: str | Path) -> str:
    """
    Load a document from disk and return its text content.

    Args:
        path: Path to the document file.

    Returns:
        Plain text content of the document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        ImportError: If a required optional dependency is not installed.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    loaders = {
        ".txt":  _load_text,
        ".md":   _load_text,
        ".rst":  _load_text,
        ".csv":  _load_csv,
        ".json": _load_json,
        ".html": _load_html,
        ".htm":  _load_html,
        ".pdf":  _load_pdf,
        ".docx": _load_docx,
    }

    if suffix not in loaders:
        supported = ", ".join(sorted(loaders.keys()))
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {supported}"
        )

    return loaders[suffix](path)


def load_text(text: str) -> str:
    """Pass-through for raw text strings."""
    return text.strip()


# ── Format-specific loaders ───────────────────────────────────────────────────

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _load_csv(path: Path) -> str:
    """Convert CSV to a markdown table for better LLM comprehension."""
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return ""

    header = rows[0]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows[1:]:
        # Pad row if shorter than header
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")

    return "\n".join(lines)


def _load_json(path: Path) -> str:
    """Load JSON and pretty-print it."""
    content = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(content)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content  # Return raw if invalid JSON


def _load_html(path: Path) -> str:
    """Strip HTML tags and return readable text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "HTML loading requires beautifulsoup4. "
            "Install it with: pip install 'llm-extract[html]'"
        )

    content = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(content, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()

    # Get text with reasonable whitespace
    text = soup.get_text(separator="\n")
    # Collapse multiple blank lines
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _load_pdf(path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "PDF loading requires pdfplumber. "
            "Install it with: pip install 'llm-extract[pdf]'"
        )

    pages = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"--- Page {page_num} ---\n{text.strip()}")

    return "\n\n".join(pages) if pages else ""


def _load_docx(path: Path) -> str:
    """Extract text from Word document."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "DOCX loading requires python-docx. "
            "Install it with: pip install 'llm-extract[docx]'"
        )

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def detect_format(path: str | Path) -> str:
    """Return the format name for a given file path."""
    suffix = Path(path).suffix.lower()
    names = {
        ".txt": "plain text", ".md": "markdown", ".rst": "reStructuredText",
        ".csv": "CSV", ".json": "JSON", ".html": "HTML", ".htm": "HTML",
        ".pdf": "PDF", ".docx": "Word document",
    }
    return names.get(suffix, f"unknown ({suffix})")
