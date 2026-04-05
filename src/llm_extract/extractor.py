"""
Core extraction engine for llm-extract.

The extraction pipeline:
1. Load document (via loader.py)
2. Chunk text if needed (via chunker.py)
3. Build extraction prompt with schema
4. Call LLM, requesting JSON with confidence scores and source snippets
5. Parse and validate response against Pydantic schema
6. Merge results across chunks if multi-chunk document
7. Return ExtractionResult
"""

from __future__ import annotations

import json
import re
from typing import Type

from pydantic import BaseModel, ValidationError

from llm_extract.chunker import chunk_text
from llm_extract.loader import load_document, load_text
from llm_extract.models import DocumentChunk, ExtractionConfig, ExtractionResult

# ── Public API ────────────────────────────────────────────────────────────────


def extract(
    source: str,
    schema: Type[BaseModel],
    config: ExtractionConfig | None = None,
    instructions: str | None = None,
) -> ExtractionResult:
    """
    Extract structured data from a document file.

    Args:
        source:       Path to document (PDF, DOCX, HTML, CSV, TXT, MD, JSON)
        schema:       Pydantic model class defining the extraction schema
        config:       Extraction configuration (optional)
        instructions: Additional extraction instructions (optional)

    Returns:
        ExtractionResult with populated data, confidence scores, and source snippets

    Example:
        from llm_extract import extract
        from pydantic import BaseModel

        class Contract(BaseModel):
            parties: list[str]
            effective_date: str
            value_gbp: float | None

        result = extract("contract.pdf", schema=Contract)
        if result.success:
            print(result.data)
            print(result.confidence)
    """
    cfg = config or ExtractionConfig()

    # Load the document
    try:
        text = load_document(source)
    except (FileNotFoundError, ValueError, ImportError) as e:
        return ExtractionResult(
            data=None,
            raw_response=str(e),
            document_path=str(source),
            fields_total=len(schema.model_fields),
        )

    result = extract_text(text, schema=schema, config=cfg, instructions=instructions)
    result.document_path = str(source)
    return result


def extract_text(
    text: str,
    schema: Type[BaseModel],
    config: ExtractionConfig | None = None,
    instructions: str | None = None,
) -> ExtractionResult:
    """
    Extract structured data from a plain text string.

    Args:
        text:         Input text to extract from
        schema:       Pydantic model class defining the extraction schema
        config:       Extraction configuration (optional)
        instructions: Additional extraction instructions (optional)

    Returns:
        ExtractionResult

    Example:
        from llm_extract import extract_text
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int
            email: str | None

        result = extract_text(
            "Dr. Sarah Chen, aged 34, can be reached at s.chen@nhs.uk",
            schema=Person,
        )
    """
    cfg = config or ExtractionConfig()
    text = load_text(text)
    fields_total = len(schema.model_fields)

    if not text:
        return ExtractionResult(
            data=None,
            raw_response="Empty input text",
            fields_total=fields_total,
        )

    chunks = chunk_text(text, chunk_size=cfg.chunk_size, overlap=cfg.overlap)

    if len(chunks) == 1:
        return _extract_single_chunk(chunks[0], schema, cfg, instructions)
    else:
        return _extract_multi_chunk(chunks, schema, cfg, instructions)


# ── Internal extraction ───────────────────────────────────────────────────────


def _extract_single_chunk(
    chunk: DocumentChunk,
    schema: Type[BaseModel],
    config: ExtractionConfig,
    instructions: str | None,
) -> ExtractionResult:
    """Extract from a single chunk of text."""
    prompt = _build_prompt(chunk.text, schema, instructions)
    raw = _call_llm(prompt, config)
    return _parse_response(raw, schema, chunks_used=1)


def _extract_multi_chunk(
    chunks: list[DocumentChunk],
    schema: Type[BaseModel],
    config: ExtractionConfig,
    instructions: str | None,
) -> ExtractionResult:
    """
    Extract from multiple chunks and merge results.
    Higher-confidence values from later chunks override lower-confidence
    values from earlier chunks.
    """
    merged_data: dict = {}
    merged_confidence: dict[str, float] = {}
    merged_sources: dict[str, str] = {}
    raw_responses = []

    for chunk in chunks:
        prompt = _build_prompt(
            chunk.text,
            schema,
            instructions,
            context=f"[Chunk {chunk.chunk_index + 1} of {chunk.total_chunks}]",
        )
        raw = _call_llm(prompt, config)
        raw_responses.append(raw)

        parsed = _parse_response(raw, schema, chunks_used=len(chunks))
        if parsed.data is None:
            continue

        chunk_data = parsed.data.model_dump()
        for field_name, value in chunk_data.items():
            if value is None:
                continue
            chunk_confidence = parsed.confidence.get(field_name, 0.5)
            existing_confidence = merged_confidence.get(field_name, 0.0)

            # Keep the value with higher confidence
            if chunk_confidence >= existing_confidence:
                merged_data[field_name] = value
                merged_confidence[field_name] = chunk_confidence
                if field_name in parsed.sources:
                    merged_sources[field_name] = parsed.sources[field_name]

    # Build final model from merged data
    fields_total = len(schema.model_fields)
    try:
        final_model = schema(**merged_data)
        fields_found = sum(1 for v in merged_data.values() if v is not None)
    except ValidationError:
        final_model = None
        fields_found = 0

    return ExtractionResult(
        data=final_model,
        confidence=merged_confidence,
        sources=merged_sources,
        raw_response="\n---\n".join(raw_responses),
        chunks_used=len(chunks),
        fields_found=fields_found,
        fields_total=fields_total,
    )


# ── Prompt building ───────────────────────────────────────────────────────────


def _build_prompt(
    text: str,
    schema: Type[BaseModel],
    instructions: str | None,
    context: str = "",
) -> str:
    """Build the extraction prompt."""
    schema_description = _describe_schema(schema)
    extra = f"\n\nAdditional instructions: {instructions}" if instructions else ""
    chunk_note = f"\n\n{context}" if context else ""

    return f"""You are a structured data extraction system. Extract information from the document below according to the schema.

SCHEMA:
{schema_description}

RULES:
1. Extract only information explicitly stated in the document. Do not infer or hallucinate.
2. For each field, provide:
   - "value": the extracted value (null if not found)
   - "confidence": float 0.0–1.0 (how confident you are this is correct)
   - "source": the exact text snippet from the document that supports this value (max 100 chars)
3. Confidence guidelines:
   - 0.95–1.0: Explicitly stated, unambiguous
   - 0.80–0.94: Clearly implied or slightly ambiguous phrasing
   - 0.60–0.79: Inferred from context, may be wrong
   - Below 0.60: Very uncertain — consider leaving as null
4. Return ONLY valid JSON. No preamble, no explanation, no markdown fences.{extra}{chunk_note}

DOCUMENT:
{text}

RESPONSE FORMAT:
{{
  "field_name": {{
    "value": <extracted value or null>,
    "confidence": <float 0.0-1.0>,
    "source": "<text snippet>"
  }},
  ...
}}"""


def _describe_schema(schema: Type[BaseModel]) -> str:
    """Generate a human-readable schema description from a Pydantic model."""
    lines = [f"Model: {schema.__name__}"]
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        description = field_info.description or ""
        required = field_info.is_required()
        req_label = "required" if required else "optional"
        desc_part = f" — {description}" if description else ""
        lines.append(f"  - {field_name} ({type_name}, {req_label}){desc_part}")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────


def _call_llm(prompt: str, config: ExtractionConfig) -> str:
    """Call the configured LLM provider and return the raw text response."""
    if config.provider == "anthropic":
        return _call_anthropic(prompt, config)
    raise ValueError(f"Unsupported provider: {config.provider}. Currently supported: 'anthropic'")


def _call_anthropic(prompt: str, config: ExtractionConfig) -> str:
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("anthropic package required. Run: pip install anthropic")

    client = Anthropic()
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Response parsing ──────────────────────────────────────────────────────────


def _parse_response(
    raw: str,
    schema: Type[BaseModel],
    chunks_used: int = 1,
) -> ExtractionResult:
    """Parse LLM JSON response into an ExtractionResult."""
    fields_total = len(schema.model_fields)

    # Strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return ExtractionResult(
            data=None,
            raw_response=raw,
            chunks_used=chunks_used,
            fields_total=fields_total,
        )

    # Extract values, confidence, sources
    extracted_values: dict = {}
    confidence_scores: dict[str, float] = {}
    source_snippets: dict[str, str] = {}

    for field_name in schema.model_fields:
        if field_name not in parsed:
            continue
        field_data = parsed[field_name]
        if not isinstance(field_data, dict):
            # Handle flat format (value only, no confidence/source)
            extracted_values[field_name] = field_data
            confidence_scores[field_name] = 0.5
            continue

        value = field_data.get("value")
        confidence = float(field_data.get("confidence", 0.5))
        source = field_data.get("source", "")

        if value is not None:
            extracted_values[field_name] = value
            confidence_scores[field_name] = max(0.0, min(1.0, confidence))
            if source:
                source_snippets[field_name] = str(source)[:200]

    # Validate against Pydantic schema
    try:
        model_instance = schema(**extracted_values)
        fields_found = sum(1 for v in extracted_values.values() if v is not None)
    except (ValidationError, TypeError):
        return ExtractionResult(
            data=None,
            confidence=confidence_scores,
            sources=source_snippets,
            raw_response=raw,
            chunks_used=chunks_used,
            fields_found=0,
            fields_total=fields_total,
        )

    return ExtractionResult(
        data=model_instance,
        confidence=confidence_scores,
        sources=source_snippets,
        raw_response=raw,
        chunks_used=chunks_used,
        fields_found=fields_found,
        fields_total=fields_total,
    )
