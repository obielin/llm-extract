# llm-extract

**Extract structured data from any document using LLMs.**

PDF, DOCX, HTML, CSV, plain text → validated Pydantic model, with per-field confidence scores and source grounding.

[![PyPI](https://img.shields.io/badge/PyPI-llm--extract-blue?style=flat-square)](https://pypi.org/project/llm-extract/)
[![Tests](https://img.shields.io/badge/Tests-42%20passing-brightgreen?style=flat-square)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/-Linda_Oraegbunam-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/linda-oraegbunam/)

---

## Why llm-extract?

[Instructor](https://github.com/567-labs/instructor) extracts from strings. [LangExtract](https://github.com/google/langextract) extracts from text with source grounding. **llm-extract** does both — and handles the document loading for you.

| Feature | Instructor | LangExtract | **llm-extract** |
|---|---|---|---|
| Pydantic schema validation | ✓ | ✗ | ✓ |
| Per-field confidence scores | ✗ | ✗ | ✓ |
| Source grounding | ✗ | ✓ | ✓ |
| Multi-format document loading | ✗ | ✗ | ✓ |
| Long document chunking | ✗ | ✓ | ✓ |
| CLI | ✗ | ✗ | ✓ |
| Provider-agnostic | ✓ | ✗ | ✓ (Anthropic; more planned) |

---

## Install

```bash
pip install llm-extract              # core (plain text, CSV, JSON)
pip install 'llm-extract[pdf]'       # + PDF support
pip install 'llm-extract[docx]'      # + Word document support
pip install 'llm-extract[html]'      # + HTML support
pip install 'llm-extract[all]'       # everything
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick Start

### Python API

```python
from llm_extract import extract
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    date: str
    invoice_number: str | None = None

result = extract("invoice.pdf", schema=Invoice)

print(result.data)
# Invoice(vendor='Acme Corp', total=1250.0, date='2026-01-15', invoice_number='INV-4521')

print(result.confidence)
# {'vendor': 0.98, 'total': 0.95, 'date': 0.99, 'invoice_number': 0.92}

print(result.sources)
# {'vendor': 'Acme Corp Ltd\n123 Business Park', 'total': 'Total Due: £1,250.00', ...}

print(result.coverage)   # 1.0  (all 4 fields found)
print(result.mean_confidence)  # 0.96
```

### From plain text

```python
from llm_extract import extract_text
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str | None = None

result = extract_text(
    "Dr. Sarah Chen, aged 34, can be reached at s.chen@nhs.uk",
    schema=Person,
)
print(result.data)
# Person(name='Dr. Sarah Chen', age=34, email='s.chen@nhs.uk')
```

### CLI

```bash
# Extract with a schema file
llm-extract invoice.pdf --schema schemas/invoice.py

# Quick extraction — no schema file needed
llm-extract contract.txt --fields "parties,effective_date,value,governing_law"

# Show where each value was found
llm-extract report.pdf --schema schemas/report.py --show-sources

# Save result as JSON
llm-extract data.csv --fields "name,address,amount" --out result.json

# List supported formats
llm-extract formats
```

---

## Schema Files

Define a Pydantic model in a `.py` file — `llm-extract` finds it automatically:

```python
# schemas/contract.py
from pydantic import BaseModel, Field
from typing import Optional

class Contract(BaseModel):
    parties: list[str] = Field(description="All parties named in the contract")
    effective_date: str = Field(description="When the contract takes effect (ISO date)")
    value_gbp: Optional[float] = Field(None, description="Contract value in GBP if stated")
    governing_law: Optional[str] = Field(None, description="Governing law jurisdiction")
    termination_notice: Optional[str] = Field(None, description="Required notice period for termination")
```

Then:
```bash
llm-extract contract.pdf --schema schemas/contract.py --show-sources
```

---

## Confidence Scores

Every extracted field gets a confidence score (0.0–1.0):

```python
result = extract("document.pdf", schema=MySchema)

# Check for low-confidence fields before using the data
low = result.low_confidence_fields(threshold=0.7)
if low:
    print(f"Review these manually: {low}")

# Filter out uncertain fields
if result.confidence.get("amount", 0) < 0.8:
    print("Amount extraction uncertain — please verify")
```

| Score | Meaning |
|---|---|
| 0.95–1.0 | Explicitly stated, unambiguous |
| 0.80–0.94 | Clearly implied, minor ambiguity |
| 0.60–0.79 | Inferred from context, may be wrong |
| Below 0.60 | Very uncertain — verify manually |

---

## Long Documents

llm-extract automatically chunks long documents and merges results. Higher-confidence values from later chunks override lower-confidence values from earlier chunks:

```python
from llm_extract import extract
from llm_extract.models import ExtractionConfig

config = ExtractionConfig(
    chunk_size=6000,   # characters per chunk (default: 8000)
    overlap=800,       # overlap between chunks (default: 500)
)

result = extract("long_report.pdf", schema=ReportSchema, config=config)
print(f"Processed {result.chunks_used} chunks")
```

---

## Supported Formats

| Format | Extensions | Extra Install |
|---|---|---|
| Plain text | `.txt` `.md` `.rst` | None |
| CSV | `.csv` | None |
| JSON | `.json` | None |
| HTML | `.html` `.htm` | `pip install 'llm-extract[html]'` |
| PDF | `.pdf` | `pip install 'llm-extract[pdf]'` |
| Word | `.docx` | `pip install 'llm-extract[docx]'` |

---

## Examples

See [`examples/`](examples/) for complete worked examples:

- [`examples/invoice_extraction.py`](examples/invoice_extraction.py) — Invoice data from PDF
- [`examples/contract_extraction.py`](examples/contract_extraction.py) — Contract terms from DOCX
- [`examples/clinical_notes.py`](examples/clinical_notes.py) — Structured data from clinical notes
- [`examples/csv_to_schema.py`](examples/csv_to_schema.py) — Typed schema from CSV data

---

## Comparison with LangExtract

Google's [LangExtract](https://github.com/google/langextract) uses few-shot examples and Gemini-specific controlled generation. llm-extract takes a different approach:

- **Schema-first**: define a Pydantic model, not examples
- **Confidence scores**: every field tells you how certain the extraction is
- **Multi-format**: load the document from disk, don't pre-process it yourself
- **Provider-agnostic**: Anthropic Claude today; OpenAI and local models planned
- **CLI included**: run extractions without writing Python

---

## Contributing

Contributions welcome — especially:
- Additional LLM providers (OpenAI, Ollama, Gemini)
- New document format loaders
- Evaluation benchmarks on real document types

---

**Linda Oraegbunam** | [LinkedIn](https://www.linkedin.com/in/linda-oraegbunam/) | [Twitter](https://twitter.com/Obie_Linda)
