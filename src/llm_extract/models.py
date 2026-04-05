"""
Core data models for llm-extract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class ExtractionConfig:
    """
    Configuration for an extraction run.

    Args:
        model:       Anthropic model to use (default: claude-opus-4-5)
        max_tokens:  Max tokens for extraction response (default: 2000)
        temperature: LLM temperature — keep low for extraction (default: 0.1)
        chunk_size:  Max characters per chunk for long documents (default: 8000)
        overlap:     Character overlap between chunks (default: 500)
        confidence_threshold: Minimum confidence to include a field (default: 0.0)
        provider:    LLM provider — currently 'anthropic' (default: 'anthropic')
    """

    model: str = "claude-opus-4-5"
    max_tokens: int = 2000
    temperature: float = 0.1
    chunk_size: int = 8000
    overlap: int = 500
    confidence_threshold: float = 0.0
    provider: str = "anthropic"


@dataclass
class ExtractionResult:
    """
    Result of a structured extraction.

    Attributes:
        data:           Populated Pydantic model instance (or None if extraction failed)
        confidence:     Per-field confidence scores (0.0–1.0)
        sources:        Per-field source text snippets showing where each value was found
        raw_response:   Raw LLM response for debugging
        document_path:  Path to the source document (if from file)
        chunks_used:    Number of text chunks processed
        fields_found:   Number of fields successfully extracted
        fields_total:   Total number of fields in the schema
    """

    data: BaseModel | None
    confidence: dict[str, float] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    raw_response: str = ""
    document_path: str | None = None
    chunks_used: int = 1
    fields_found: int = 0
    fields_total: int = 0

    @property
    def success(self) -> bool:
        """True if extraction produced a populated model."""
        return self.data is not None

    @property
    def coverage(self) -> float:
        """Fraction of schema fields successfully extracted (0.0–1.0)."""
        if self.fields_total == 0:
            return 0.0
        return self.fields_found / self.fields_total

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across all extracted fields."""
        if not self.confidence:
            return 0.0
        return sum(self.confidence.values()) / len(self.confidence)

    def low_confidence_fields(self, threshold: float = 0.7) -> list[str]:
        """Return field names with confidence below the threshold."""
        return [f for f, c in self.confidence.items() if c < threshold]

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"ExtractionResult({status} coverage={self.coverage:.0%} "
            f"mean_confidence={self.mean_confidence:.2f} "
            f"fields={self.fields_found}/{self.fields_total})"
        )


@dataclass
class DocumentChunk:
    """A chunk of document text with positional metadata."""

    text: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    page_number: int | None = None
