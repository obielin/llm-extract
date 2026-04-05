"""
llm-extract
===========
Extract structured data from any document using LLMs.
Supports PDF, DOCX, HTML, CSV, and plain text with Pydantic
schema validation, per-field confidence scores, and source grounding.

Quick start:
    from llm_extract import extract
    from pydantic import BaseModel

    class Invoice(BaseModel):
        vendor: str
        total: float
        date: str

    result = extract("invoice.pdf", schema=Invoice)
    print(result.data)          # Invoice(vendor='Acme', total=1250.0, date='2026-01-15')
    print(result.confidence)    # {'vendor': 0.98, 'total': 0.95, 'date': 0.99}
    print(result.sources)       # {'vendor': 'Acme Corp\\nInvoice #1234', ...}
"""

from llm_extract.extractor import extract, extract_text
from llm_extract.loader import load_document
from llm_extract.models import ExtractionConfig, ExtractionResult

__version__ = "1.0.0"
__author__ = "Linda Oraegbunam"
__all__ = [
    "extract",
    "extract_text",
    "load_document",
    "ExtractionResult",
    "ExtractionConfig",
]
